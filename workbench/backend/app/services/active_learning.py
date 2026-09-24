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


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(config.ROOT))
    except ValueError:
        return str(path)


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
    """Build a replay-mixed AL dataset with group-isolated regression holdout."""
    seed = seed if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    val_ratio = val_ratio if val_ratio is not None else config.VAL_RATIO
    if not 0 < val_ratio < 0.5:
        raise ValueError("val_ratio must be between 0 and 0.5")

    approved = case_svc.get_approved_cases(conn)
    if not approved:
        raise ValueError("no approved AL cases to build dataset from")
    original_rows = conn.execute("SELECT * FROM examples WHERE status = 'approved'").fetchall()
    original_hashes = {row["content_hash"] for row in original_rows}

    al_examples: list[dict[str, Any]] = []
    seen_hashes = set(original_hashes)
    for case in approved:
        edited_json = case.get("edited_example_json")
        if not edited_json:
            continue
        try:
            example = json.loads(edited_json) if isinstance(edited_json, str) else dict(edited_json)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(example, dict) or not example.get("messages"):
            continue
        content_h = _compute_content_hash(example)
        if content_h in seen_hashes:
            continue
        seen_hashes.add(content_h)
        example["source"] = "active_learning"
        example["al_case_id"] = case["case_id"]
        example["al_session_id"] = case["session_id"]
        # Keep all examples from the same project/session in one split.
        example["group_id"] = example.get("group_id") or case.get("project") or f"al/session/{case['session_id']}"
        example["content_hash"] = content_h
        al_examples.append(example)

    original_examples = [{
        "id": row["id"],
        "category": row["category"],
        "messages": json.loads(row["messages_json"]),
        "tools": json.loads(row["tools_json"]) if row["tools_json"] else None,
        "source": row["source"],
        "group_id": row["group_id"],
        "status": row["status"],
        "content_hash": row["content_hash"],
    } for row in original_rows]

    if categories:
        original_examples = [e for e in original_examples if e["category"] in categories]
        al_examples = [e for e in al_examples if e["category"] in categories]
    if not original_examples:
        raise ValueError("at least one approved original example is required for replay; AL data is capped at 30%")

    # Shared validator pass again at dataset-build time; invalid legacy examples
    # cannot sneak into training just because their DB status says approved.
    from backend.validators.base import Example as FrameworkExample
    from backend.validators.registry import make_context, validate_example
    peers = []
    for item in original_examples:
        peers.append(FrameworkExample(
            id=item["id"], category=item["category"], messages=item["messages"],
            tools=item.get("tools"), source=item.get("source", "manual"),
            group_id=item.get("group_id"), status="approved", content_hash=item.get("content_hash"),
        ))
    ctx = make_context(peers=peers)
    def _valid(item: dict[str, Any]) -> bool:
        framework = FrameworkExample(
            id=item.get("id", "al-candidate"), category=item["category"],
            messages=item["messages"], tools=item.get("tools"),
            source=item.get("source", "active_learning"), group_id=item.get("group_id"),
            status="approved", content_hash=item.get("content_hash"),
        )
        return validate_example(framework, ctx).ok
    original_examples = [item for item in original_examples if _valid(item)]
    al_examples = [item for item in al_examples if _valid(item)]
    if not original_examples:
        raise ValueError("no approved original examples pass the shared validators")
    if not al_examples:
        raise ValueError("no new unique AL examples pass the shared validators")

    # Frozen-suite contamination screening, matching T2.8 / plan 04 §4.
    from backend.pipeline.validator import ContaminationChecker
    checker = ContaminationChecker.from_eval_dir(threshold=config.EVAL_OVERLAP_THRESHOLD)
    clean_al: list[dict[str, Any]] = []
    contamination_dropped = []
    for item in al_examples:
        result = checker.check(item)
        if result.has_contamination:
            contamination_dropped.append({"case_id": item.get("al_case_id"), "overlaps": result.overlaps})
        else:
            clean_al.append(item)
    al_examples = clean_al
    if not al_examples:
        raise ValueError("all approved AL examples overlap the frozen eval suites")

    # At most 30% of the full selected dataset may come from Active Learning.
    max_al = int(len(original_examples) * AL_MAX_RATIO / (1 - AL_MAX_RATIO))
    if max_al < 1:
        raise ValueError("not enough original replay examples to include AL data under the 30% cap")
    al_examples.sort(key=lambda e: (str(e.get("al_case_id", "")), str(e.get("id", ""))))
    al_examples = al_examples[:max_al]

    for item in al_examples:
        item.setdefault("id", ids.example_id())
    all_examples = original_examples + al_examples

    train, val, regression = _split_with_regression(all_examples, val_ratio, seed)
    if not train or not val:
        raise ValueError("need at least two independent approved original groups to create non-empty train and validation splits")
    ds_id = _next_dataset_version_id(conn)
    out_dir = config.DATASETS_DIR / ds_id
    out_dir.mkdir(parents=True, exist_ok=False)
    train_path, val_path = out_dir / "train.jsonl", out_dir / "val.jsonl"
    regression_path = out_dir / "regression.jsonl"
    _write_examples_jsonl(train_path, train)
    _write_examples_jsonl(val_path, val)
    if regression:
        _write_examples_jsonl(regression_path, regression)

    used_case_ids = sorted({e["al_case_id"] for e in train + val + regression if e.get("al_case_id")})
    manifest = _build_manifest(
        ds_id, name, seed, val_ratio, train, val, regression, len(used_case_ids)
    )
    manifest["selection"] = {
        "approved_cases_seen": len(approved),
        "included_case_ids": used_case_ids,
        "contamination_dropped": contamination_dropped,
    }
    manifest["files"] = {
        "train": {"path": _rel(train_path), "sha256": _sha256_file(train_path), "lines": len(train)},
        "val": {"path": _rel(val_path), "sha256": _sha256_file(val_path), "lines": len(val)},
    }
    if regression:
        manifest["files"]["regression"] = {
            "path": _rel(regression_path),
            "sha256": _sha256_file(regression_path), "lines": len(regression),
        }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    all_ids = [e["id"] for e in train + val + regression]
    conn.execute(
        """INSERT INTO dataset_versions
           (id, name, example_ids_json, split_json, seed, manifest_json, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (
            ds_id, name, json.dumps(all_ids, ensure_ascii=False),
            json.dumps({
                "train": [e["id"] for e in train],
                "val": [e["id"] for e in val],
                "regression": [e["id"] for e in regression],
            }, ensure_ascii=False),
            seed, json.dumps(manifest, ensure_ascii=False), _now(),
        ),
    )
    conn.commit()

    # Mark only examples actually emitted to an artifact as merged. Capped or
    # contaminated cases remain approved for the next round.
    for case_id in used_case_ids:
        conn.execute(
            "UPDATE case_queue SET status = ?, updated_at = ? WHERE case_id = ? AND status = ?",
            ("merged", _now(), case_id, case_svc.CASE_STATUS_APPROVED),
        )
    conn.commit()
    return {"id": ds_id, "name": name, "seed": seed, "created_at": _now(), "manifest": manifest}

def _split_with_regression(
    examples: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split originals group-aware and hold out whole AL session/project groups.

    An AL session/project can never appear in both training and the regression
    file, even when several approved cases came from the same group.
    """
    rng = random.Random(seed)
    al_examples = [e for e in examples if e.get("source") == "active_learning"]
    original_examples = [e for e in examples if e.get("source") != "active_learning"]

    by_group: dict[str, list[dict]] = defaultdict(list)
    for ex in original_examples:
        group = ex.get("group_id") or f"__solo__{ex['id']}"
        by_group[str(group)].append(ex)
    group_keys = sorted(by_group)
    rng.shuffle(group_keys)
    target_val = max(1, round(len(original_examples) * val_ratio)) if len(original_examples) > 1 else 0
    train_orig: list[dict] = []
    val_orig: list[dict] = []
    for group in group_keys:
        members = by_group[group]
        if len(group_keys) > 1 and len(val_orig) < target_val:
            val_orig.extend(members)
        else:
            train_orig.extend(members)

    al_by_group: dict[str, list[dict]] = defaultdict(list)
    for ex in al_examples:
        group = ex.get("group_id") or ex.get("al_session_id") or f"__solo__{ex['id']}"
        al_by_group[str(group)].append(ex)
    al_groups = sorted(al_by_group)
    regression_groups: set[str] = set()
    if len(al_groups) > 1:
        holdout_rng = random.Random(seed + 1)
        holdout_rng.shuffle(al_groups)
        target_regression = max(1, round(len(al_examples) * 0.2))
        held_count = 0
        # Keep at least one complete group available for training.
        for group in al_groups[:-1]:
            if held_count >= target_regression:
                break
            regression_groups.add(group)
            held_count += len(al_by_group[group])

    regression = [e for group in sorted(regression_groups) for e in al_by_group[group]]
    train_al = [e for group in sorted(al_by_group) if group not in regression_groups for e in al_by_group[group]]
    return train_orig + train_al, val_orig, regression


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

    from backend.adapters.registry import get_adapter

    adapter = get_adapter(config.ACTIVE_ADAPTER)
    return {
        "dataset_id": ds_id,
        "name": name,
        "created_at": _now(),
        "adapter": adapter.name,
        "adapter_hf_repo": adapter.hf_repo,
        "template_hashes": {
            "default": adapter.template_hash("default"),
            "tool_use": adapter.template_hash("tool_use") if adapter.tool_use_template_file else None,
        },
        "max_seq_len": config.MAX_SEQ_LEN,
        "seed": seed,
        "val_ratio": val_ratio,
        "active_learning": {
            "al_case_count": al_case_count,
            "al_example_count": len([e for e in train + val + (regression or []) if e.get("source") == "active_learning"]),
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
        "split_policy": "group-aware originals; whole AL session/project groups held out for regression",
    }
