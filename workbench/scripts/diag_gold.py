#!/usr/bin/env python3
"""Diagnose remaining gold set errors."""
import json

# Tool C5 errors
print("=== TOOL C5 ERRORS ===")
with open("data/gold/tool.jsonl") as f:
    for line in f:
        ex = json.loads(line)
        if ex["id"] in ("ex_gold_tool_028", "ex_gold_tool_029", "ex_gold_tool_030"):
            for m in ex["messages"]:
                content = str(m.get("content", "") or "")
                if "password" in content.lower() or "DB_PASSWORD" in content:
                    print(ex["id"], m["role"], ":", content[:200])
            print()

# Sec S4 errors
print("=== SEC S4 ERRORS ===")
import re
hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
with open("data/gold/sec.jsonl") as f:
    for line in f:
        ex = json.loads(line)
        if ex["id"] in ("ex_gold_sec_023", "ex_gold_sec_024", "ex_gold_sec_025", "ex_gold_sec_031", "ex_gold_sec_032"):
            # Get diff text
            diff_text = ""
            for m in ex["messages"]:
                if m["role"] in ("user", "tool"):
                    diff_text += str(m.get("content", "") or "") + "\n"
            
            valid = set()
            current = None
            for dl in diff_text.splitlines():
                hm = hunk_re.match(dl)
                if hm:
                    current = int(hm.group(1))
                    continue
                if current is None:
                    continue
                if dl.startswith("+++") or dl.startswith("---"):
                    continue
                if dl.startswith("+"):
                    valid.add(current)
                    current += 1
                elif dl.startswith("-"):
                    continue
                else:
                    valid.add(current)
                    current += 1
            
            # Get issue lines
            for m in ex["messages"]:
                if m["role"] == "assistant":
                    content = str(m.get("content", "") or "")
                    try:
                        obj = json.loads(content)
                        if "issues" in obj:
                            for iss in obj["issues"]:
                                print("%s: issue.line=%d, valid_lines=%s" % (ex["id"], iss["line"], sorted(valid)[:10]))
                    except:
                        pass
            # Print the diff
            print("%s diff preview:" % ex["id"])
            for dl in diff_text.splitlines()[:15]:
                print("  ", dl)
            print()
