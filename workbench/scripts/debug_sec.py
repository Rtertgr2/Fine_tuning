#!/usr/bin/env python3
"""Debug exactly what validator sees for sec S4."""
import json
from backend.validators.base import Example
from backend.validators.sec_rules import extract_diff_line_numbers, _diff_text
from backend.adapters.registry import get_adapter

adapter = get_adapter("hermes2pro-llama3-8b")

for exid in ["ex_gold_sec_023", "ex_gold_sec_031"]:
    with open("data/gold/sec.jsonl") as f:
        for line in f:
            ex = json.loads(line)
            if ex["id"] != exid:
                continue
            
            # Build Example
            e = Example(
                id=ex["id"],
                category=ex["category"],
                messages=ex["messages"],
                tools=ex.get("tools"),
            )
            
            # Get diff text
            diff_text = _diff_text(e)
            print("=== %s diff_text ===" % exid)
            for i, dl in enumerate(diff_text.split("\n")[:10]):
                print("  %d: %r" % (i, dl))
            
            # Parse valid lines
            valid = extract_diff_line_numbers(diff_text)
            print("valid lines:", sorted(valid) if valid else None)
            
            # Get issue line
            for m in ex["messages"]:
                if m["role"] == "assistant":
                    try:
                        obj = json.loads(str(m["content"]))
                        for iss in obj.get("issues", []):
                            print("issue.line:", iss["line"], "in valid?", iss["line"] in (valid or set()))
                    except:
                        pass
            print()
