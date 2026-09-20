"""Dashboard stats (T1.9): progress per category, validation pass rates,
sec PASS/REJECT share, token-length histogram."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from typing import Any

from backend.app.services import examples as ex_service
from backend.validators.base import parse_final_json
from backend.validators.registry import make_context, validate_example

_TOKEN_BUCKETS = [512, 1024, 2048, 4096, 8192, 16384]


def compute_stats(conn: sqlite3.Connection, adapter_name: str | None = None) -> dict[str, Any]:
    rows = conn.execute("SELECT * FROM examples ORDER BY id").fetchall()
    fw = [ex_service.row_to_framework(r) for r in rows]

    by_category: dict[str, int] = defaultdict(int)
    by_status: dict[str, int] = defaultdict(int)
    by_source: dict[str, int] = defaultdict(int)
    for ex in fw:
        by_category[ex.category] += 1
        by_status[ex.status] += 1
        by_source[ex.source] += 1

    ctx = make_context(peers=fw, adapter_name=adapter_name)
    pass_by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "fail": 0})
    err_codes: dict[str, int] = defaultdict(int)
    warn_codes: dict[str, int] = defaultdict(int)
    for ex in fw:
        report = validate_example(ex, ctx)
        key = "pass" if report.ok else "fail"
        pass_by_cat[ex.category][key] += 1
        for viol in report.violations:
            (err_codes if viol.level == "err" else warn_codes)[viol.code] += 1

    sec_status: dict[str, int] = defaultdict(int)
    for ex in fw:
        if ex.category != "sec":
            continue
        obj, error = parse_final_json(ex)
        if error or obj is None:
            sec_status["unparsed"] += 1
            continue
        status = obj.get("status")
        sec_status[str(status)] += 1

    histogram: dict[str, int] = {f"<= {_TOKEN_BUCKETS[0]}": 0}
    for lo, hi in zip(_TOKEN_BUCKETS, _TOKEN_BUCKETS[1:]):
        histogram[f"{lo + 1}–{hi}"] = 0
    histogram[f"> {_TOKEN_BUCKETS[-1]}"] = 0
    unknown_tokens = 0
    for row in rows:
        count = row["token_count"]
        if count is None:
            unknown_tokens += 1
            continue
        placed = False
        if count <= _TOKEN_BUCKETS[0]:
            histogram[f"<= {_TOKEN_BUCKETS[0]}"] += 1
            placed = True
        else:
            for lo, hi in zip(_TOKEN_BUCKETS, _TOKEN_BUCKETS[1:]):
                if lo < count <= hi:
                    histogram[f"{lo + 1}–{hi}"] += 1
                    placed = True
                    break
        if not placed:
            histogram[f"> {_TOKEN_BUCKETS[-1]}"] += 1

    return {
        "total": len(fw),
        "by_category": dict(sorted(by_category.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_source": dict(sorted(by_source.items())),
        "validation": {
            "by_category": {k: dict(v) for k, v in sorted(pass_by_cat.items())},
            "err_counts_by_code": dict(sorted(err_codes.items())),
            "warn_counts_by_code": dict(sorted(warn_codes.items())),
            "tokenizer_available": ctx.tokenizer_available,
        },
        "sec": {"status_counts": dict(sorted(sec_status.items()))},
        "token_histogram": histogram,
        "token_count_unknown": unknown_tokens,
    }
