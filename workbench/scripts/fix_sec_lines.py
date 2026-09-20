#!/usr/bin/env python3
"""Fix S4 line numbers in new sec examples."""
import json
from pathlib import Path

GOLD = Path("data/gold")
path = GOLD / "sec.jsonl"
lines = path.read_text().splitlines()
fixed = []
for line in lines:
    ex = json.loads(line)
    if ex["id"].startswith("ex_gold_sec_new_"):
        for m in ex["messages"]:
            if m["role"] == "assistant":
                try:
                    obj = json.loads(str(m["content"]))
                    if "issues" in obj:
                        for iss in obj["issues"]:
                            # Fix: subtract 1 from line numbers that are off-by-one
                            if iss["line"] > 1:
                                iss["line"] = iss["line"] - 1
                        m["content"] = json.dumps(obj, ensure_ascii=False)
                except:
                    pass
    fixed.append(json.dumps(ex, ensure_ascii=False))
path.write_text("\n".join(fixed) + "\n")
print("Fixed line numbers in new sec examples")
