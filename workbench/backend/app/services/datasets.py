"""Dataset versioning: group-aware stratified split + manifest (T1.8).

Rules honoured here:
- approved examples only, and only those passing every err-level rule
- examples sharing a group_id never straddle train/val
- split is stratified by category
- the seed is recorded so the same split can be reproduced
- every artifact is hashed into manifest.json
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
from collections import defaultdict
from typing import Any, Sequence

from backend.adapters.registry import get_adapter
from backend.app import config
from backend.app.services import examples as ex_service
from backend.validators.base import Example as FwExample
from backend.validators.registry import make_context, validate_example

_VERSION_RE = re.compile(r"^v(\d+)$")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def next_version_id(conn: sqlite3.Connection) -> str:
    top = 0
    for row in conn.execute("SELECT id FROM dataset_versions").fetchall():
        m = _VERSION_RE.match(row["id"])
        if m:
            top = max(top, int(m.group(1)))
    return f"v{top + 1:04d}"


def split_examples(
    examples: Sequence[FwExample], val_ratio: float, seed: int
) -> tuple[list[FwExample], list[FwExample]]:
    """Group-aware, category-stratified, deterministic split."""
    rng = random.Random(seed)

    by_cat: dict[str, dict[str, list[FwExample]]] = defaultdict(lambda: defaultdict(list))
    for ex in examples:
        group_key = ex.group_id or f"__solo__{ex.id}"
        by_cat[ex.category][group_key].append(ex)

    train: list[FwExample] = []
    val: list[FwExample] = []
    for category in sorted(by_cat):
        groups = by_cat[category]
        group_keys = sorted(groups)
        rng.shuffle(group_keys)
        total = sum(len(groups[g]) for g in group_keys)
        target_val = max(1, round(total * val_ratio)) if total > 1 else 0
        val_count = 0
        for gk in group_keys:
            members = groups[gk]
            if len(group_keys) > 1 and val_count < target_val:
                val.extend(members)
                val_count += len(members)
            else:
                train.extend(members)
    train.sort(key=lambda e: e.id)
    val.sort(key=lambda e: e.id)
    return train, val


def _counts(examples: Sequence[FwExample]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for ex in examples:
        out[ex.category] += 1
    return dict(sorted(out.items()))


def _write_jsonl(path, examples: Sequence[FwExample]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            payload = {
                "id": ex.id,
                "category": ex.category,
                "messages": ex.messages,
                "tools": ex.tools,
                "meta": {
                    "source": ex.source,
                    "group": ex.group_id,
                    "status": ex.status,
                    "content_hash": ex.content_hash,
                },
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _rel(path) -> str:
    try:
        return str(path.relative_to(config.ROOT))
    except ValueError:
        return str(path)


def build_dataset(
    conn: sqlite3.Connection,
    name: str | None = None,
    seed: int | None = None,
    val_ratio: float | None = None,
    categories: list[str] | None = None,
    adapter_name: str | None = None,
) -> dict[str, Any]:
    adapter = get_adapter(adapter_name)
    ratio = val_ratio if val_ratio is not None else config.VAL_RATIO
    use_seed = seed if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)

    approved = ex_service.all_framework(conn, status="approved")
    if categories:
        approved = [e for e in approved if e.category in categories]
    if not approved:
        raise ValueError("no approved examples to build a dataset from")

    ctx = make_context(peers=approved, adapter_name=adapter_name)
    invalid: list[str] = []
    valid: list[FwExample] = []
    err_codes: dict[str, int] = defaultdict(int)
    warn_codes: dict[str, int] = defaultdict(int)
    for ex in approved:
        report = validate_example(ex, ctx)
        if report.ok:
            valid.append(ex)
        else:
            invalid.append(ex.id)
        for viol in report.violations:
            if viol.level == "err":
                err_codes[viol.code] += 1
            else:
                warn_codes[viol.code] += 1

    if not valid:
        raise ValueError("every approved example failed err-level validation")

    train, val = split_examples(valid, ratio, use_seed)

    ds_id = next_version_id(conn)
    out_dir = config.DATASETS_DIR / ds_id
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(val_path, val)

    ts = ex_service.now_iso()
    manifest: dict[str, Any] = {
        "dataset_id": ds_id,
        "name": name,
        "created_at": ts,
        "adapter": adapter.name,
        "adapter_hf_repo": adapter.hf_repo,
        "template_hashes": {
            "default": adapter.template_hash("default"),
            "tool_use": adapter.template_hash("tool_use")
            if adapter.tool_use_template_file
            else None,
        },
        "seed": use_seed,
        "val_ratio": ratio,
        "max_seq_len": config.MAX_SEQ_LEN,
        "counts": {
            "train": _counts(train),
            "val": _counts(val),
            "train_total": len(train),
            "val_total": len(val),
            "total": len(train) + len(val),
        },
        "files": {
            "train": {
                "path": _rel(train_path),
                "sha256": sha256_file(train_path),
                "lines": len(train),
            },
            "val": {
                "path": _rel(val_path),
                "sha256": sha256_file(val_path),
                "lines": len(val),
            },
        },
        "selection": {
            "approved": len(approved),
            "excluded_invalid": invalid,
            "excluded_invalid_count": len(invalid),
        },
        "validation": {
            "err_counts_by_code": dict(sorted(err_codes.items())),
            "warn_counts_by_code": dict(sorted(warn_codes.items())),
            "tokenizer_available": ctx.tokenizer_available,
        },
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    split_json = {
        "train": [e.id for e in train],
        "val": [e.id for e in val],
    }
    conn.execute(
        """INSERT INTO dataset_versions
           (id, name, example_ids_json, split_json, seed, manifest_json, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (
            ds_id,
            name,
            json.dumps([e.id for e in [*train, *val]], ensure_ascii=False),
            json.dumps(split_json, ensure_ascii=False),
            use_seed,
            json.dumps(manifest, ensure_ascii=False),
            ts,
        ),
    )
    conn.commit()

    return {
        "id": ds_id,
        "name": name,
        "seed": use_seed,
        "created_at": ts,
        "manifest": manifest,
    }


def get_dataset(conn: sqlite3.Connection, ds_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM dataset_versions WHERE id = ?", (ds_id,)).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "seed": row["seed"],
        "created_at": row["created_at"],
        "manifest": json.loads(row["manifest_json"]),
    }


def list_datasets(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM dataset_versions ORDER BY id").fetchall()
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "seed": r["seed"],
            "created_at": r["created_at"],
            "manifest": json.loads(r["manifest_json"]),
        }
        for r in rows
    ]


def dataset_dir(ds_id: str):
    return config.DATASETS_DIR / ds_id
