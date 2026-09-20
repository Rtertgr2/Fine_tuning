#!/usr/bin/env python3
"""Validate all gold set examples against validators."""
import json
from backend.validators import registry
from backend.validators.base import Example, ValidationContext
from backend.adapters.registry import get_adapter

adapter = get_adapter("hermes2pro-llama3-8b")
ctx = ValidationContext(adapter=adapter)

categories = ["tool", "loop", "plan", "sec"]
total_err = 0
total_warn = 0
for cat in categories:
    path = f"data/gold/{cat}.jsonl"
    examples = []
    with open(path) as f:
        for line in f:
            raw = json.loads(line)
            ex = Example(
                id=raw["id"],
                category=raw["category"],
                messages=raw["messages"],
                tools=raw.get("tools"),
                source=raw.get("meta", {}).get("source", "manual"),
                group_id=raw.get("meta", {}).get("group"),
                status=raw.get("meta", {}).get("status", "draft"),
            )
            examples.append(ex)

    err_count = 0
    warn_count = 0
    first_errors = []
    for ex in examples:
        report = registry.validate_example(ex, ctx)
        for v in report.violations:
            if v.level == "err":
                err_count += 1
                if len(first_errors) < 5:
                    first_errors.append("  [%s] %s: %s" % (ex.id, v.code, v.message))
            elif v.level == "warn":
                warn_count += 1

    ds_report = registry.validate_dataset(examples, ctx)
    ds_errs = sum(1 for v in ds_report.violations if v.level == "err")
    ds_warns = sum(1 for v in ds_report.violations if v.level == "warn")

    total_err += err_count + ds_errs
    total_warn += warn_count + ds_warns

    status = "PASS" if (err_count + ds_errs) == 0 else "FAIL"
    print("%s: %d examples | %s | per-ex: %d err / %d warn | dataset: %d err / %d warn" % (
        cat, len(examples), status, err_count, warn_count, ds_errs, ds_warns))
    for e in first_errors:
        print(e)

print("")
print("TOTAL: %d err, %d warn" % (total_err, total_warn))
print("ALL ERR PASS: %s" % (total_err == 0))
