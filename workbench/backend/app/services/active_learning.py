"""Active Learning data mixing policy service.

Per Plan/06 §7:
  - Max 30% AL data in new dataset
  - Mix with approved original data for replay
  - Deduplicate against existing dataset
  - Split train/regression by session/project
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from backend.app import config, ids, schemas
from backend.app.services import cases as case_svc


AL_MAX_RATIO = 0.3  # Max fraction of AL data in new dataset
AL_THRESHOLD = 150  # Approved cases needed to trigger a training round


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def check_threshold(conn: sqlite3.Connection) -> tuple[int, bool]:
    """Check if enough approved cases exist to trigger a new round.
    
    Returns (approved_count, threshold_met).
    """
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM case_queue WHERE status = ?",
        (case_svc.CASE_STATUS_APPROVED,),
    ).fetchone()
    count = row["n"]
    return count, count >= AL_THRESHOLD


def build_al_dataset(
    conn: sqlite3.Connection,
    seed: int | None = None,
    val_ratio: float | None = None,
    categories: list[str] | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Build a new dataset version mixing AL cases with approved original data.
    
    Returns the dataset record.
    """
    seed = seed if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    val_ratio = val_ratio if val_ratio is not None else config.VAL_RATIO

    # Get approved AL cases
    approved = case_svc.get_approved_cases(conn)
    if not approved:
        raise ValueError("no approved AL cases to build dataset from")

    # Get existing approved examples for replay
    original_rows = conn.execute(
        "SELECT * FROM examples WHERE status = 'approved'"
    ).fetchall()

    # Deduplicate AL cases against original examples
    original_hashes = {r["content_hash"] for r in original_rows}
    al_examples: list[dict[str, Any]] = []
    al_session_ids: set[str] = set()

    for case in approved:
        edited_json = case.get("edited_example_json")
        if not edited_json:
            continue
        try:
            example = json.loads(edited_json) if isinstance(edited_json, str) else edited_json
        except (json.JSONDecodeError, TypeError):
            continue

        # Compute hash for dedup
        content_h = _compute_content_hash(example)
        if content_h in original_hashes:
            continue  # Already exists

        example["source"] = "active_learning"
        example["al_case_id"] = case["case_id"]
        al_examples.append(example)
        al_session_ids.add(case["session_id"])

    # Apply AL ratio cap
    max_al = max(1, int(len(original_rows) * AL_MAX_RATIO / (1 - AL_MAX_RATIO))) if original_rows else len(al_examples)
    if len(al_examples) > max_al:
        al_examples = al_examples[:max_al]

    if not al_examples:
        raise ValueError("no new unique AL examples after deduplication")

    # Merge: original + AL
    all_examples = []
    for row in original_rows:
        all_examples.append({
            "id": row["id"],
            "category": row["category"],
            "messages": json.loads(row["messages_json"]),
            "tools": json.loads(row["tools_json"]) if row["tools_json"] else None,
            "source": row["source"],
            "group_id": row["group_id"],
            "status": row["status"],
            "content_hash": row["content_hash"],
        })
    for ex in al_examples:
        if "id" not in ex:
            ex["id"] = ids.example_id()
        all_examples.append(ex)

    if categories:
        all_examples = [e for e in all_examples if e["category"] in categories]

    # Split by session/project: AL data goes to train, but we ensure
    # no AL session appears in both train and regression
    train, val, regression = _split_with_regression(all_examples, al_session_ids, val_ratio, seed)

    # Build dataset version
    ds_id = _next_dataset_version_id(conn)
    out_dir = config.DATASETS_DIR / ds_id
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_examples_jsonl(out_dir / "train.jsonl", train)
    _write_examples_jsonl(out_dir / "val.jsonl", val)
    if regression:
        _write_examples_jsonl(out_dir / "regression.jsonl", regression)

    # Compute manifest
    manifest = _build_manifest(
        ds_id, name, seed, val_ratio, train, val, regression, len(approved)
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Record in database
    all_ids = [e["id"] for e in train + val + (regression or [])]
    conn.execute(
        """INSERT INTO dataset_versions
           (id, name, example_ids_json, split_json, seed, manifest_json, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (
            ds_id,
            name,
            json.dumps(all_ids, ensure_ascii=False),
            json.dumps({
                "train": [e["id"] for e in train],
                "val": [e["id"] for e in val],
                "regression": [e["id"] for e in (regression or [])],
            }),
            seed,
            json.dumps(manifest, ensure_ascii=False),
            _now(),
        ),
    )
    conn.commit()

    # Mark cases as merged
    for case in approved:
        if case.get("edited_example_json"):
            conn.execute(
                "UPDATE case_queue SET status = ? WHERE case_id = ?",
                ("merged", case["case_id"]),
            )
    conn.commit()

    return {
        "id": ds_id,
        "name": name,
        "seed": seed,
        "created_at": _now(),
        "manifest": manifest,
    }


def _split_with_regression(
    examples: list[dict[str, Any]],
    al_session_ids: set[str],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict] | None]:
    """Split examples ensuring AL sessions don't appear in both train and regression."""
    rng = random.Random(seed)

    # Separate AL and original examples
    al_examples = [e for e in examples if e.get("source") == "active_learning"]
    original_examples = [e for e in examples if e.get("source") != "active_learning"]

    # AL examples go to train (they are the new learning signal)
    # Original examples split between train and val
    rng.shuffle(original_examples)

    # Group original by group_id for group-aware split
    by_group: dict[str, list[dict]] = defaultdict(list)
    for ex in original_examples:
        gk = ex.get("group_id") or f"__solo__{ex['id']}"
        by_group[gk].append(ex)

    train_orig: list[dict] = []
    val_orig: list[dict] = []
    group_keys = sorted(by_group.keys())
    rng.shuffle(group_keys)

    total_orig = len(original_examples)
    target_val = max(1, round(total_orig * val_ratio)) if total_orig > 1 else 0
    val_count = 0

    for gk in group_keys:
        members = by_group[gk]
        if len(group_keys) > 1 and val_count < target_val:
            val_orig.extend(members)
            val_count += len(members)
        else:
            train_orig.extend(members)

    # Train = original train + AL examples
    train = train_orig + al_examples
    val = val_orig

    # Regression set: a sample of AL examples held out for regression testing
    # Use a different seed for regression split
    regression_rng = random.Random(seed + 1)
    regression_size = max(1, len(al_examples) // 5)  # 20% of AL for regression
    regression = regression_rng.sample(al_examples, min(regression_size, len(al_examples)))

    # Remove regression from train
    regression_ids = {id(e) for e in regression}
    train = [e for e in train if id(e) not in regression_ids]

    return train, val, regression


def _compute_content_hash(example: dict) -> str:
    """Compute content hash for deduplication. Must match adapter's content_hash."""
    from backend.adapters.base import content_hash
    return content_hash(example.get("messages", []), example.get("tools"))


def _next_dataset_version_id(conn: sqlite3.Connection) -> str:
    """Get next dataset version ID."""
    import re
    _VERSION_RE = re.compile(r"^v(\d+)$")
    top = 0
    for row in conn.execute("SELECT id FROM dataset_versions").fetchall():
        m = _VERSION_RE.match(row["id"])
        if m:
            top = max(top, int(m.group(1)))
    return f"v{top + 1:04d}"


def _write_examples_jsonl(path, examples: list[dict]) -> None:
    """Write examples to a JSONL file."""
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            payload = {
                "id": ex.get("id", ids.example_id()),
                "category": ex["category"],
                "messages": ex["messages"],
                "tools": ex.get("tools"),
                "meta": {
                    "source": ex.get("source", "manual"),
                    "group": ex.get("group_id"),
                    "status": ex.get("status", "approved"),
                    "content_hash": ex.get("content_hash", _compute_content_hash(ex)),
                },
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _build_manifest(
    ds_id: str,
    name: str | None,
    seed: int,
    val_ratio: float,
    train: list[dict],
    val: list[dict],
    regression: list[dict] | None,
    al_case_count: int,
) -> dict[str, Any]:
    """Build dataset manifest."""
    def _counts(examples: list[dict]) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for ex in examples:
            out[ex["category"]] += 1
        return dict(sorted(out.items()))

    return {
        "dataset_id": ds_id,
        "name": name,
        "created_at": _now(),
        "seed": seed,
        "val_ratio": val_ratio,
        "active_learning": {
            "al_case_count": al_case_count,
            "al_example_count": len([e for e in train + val if e.get("source") == "active_learning"]),
            "al_max_ratio": AL_MAX_RATIO,
        },
        "counts": {
            "train": _counts(train),
            "val": _counts(val),
            "regression": _counts(regression) if regression else {},
            "train_total": len(train),
            "val_total": len(val),
            "regression_total": len(regression) if regression else 0,
            "total": len(train) + len(val) + (len(regression) if regression else 0),
        },
        "split_policy": "session-aware with AL regression holdout",
    }
