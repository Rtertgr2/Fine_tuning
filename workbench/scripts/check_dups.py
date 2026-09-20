#!/usr/bin/env python3
"""Check content_hash duplicates in gold set."""
import json, hashlib
from pathlib import Path

GOLD = Path("data/gold")

for cat in ["tool", "loop", "plan", "sec"]:
    path = GOLD / ("%s.jsonl" % cat)
    hashes = {}
    dups = 0
    with open(path) as f:
        for i, line in enumerate(f, 1):
            ex = json.loads(line)
            # Normalize: sort keys, compact JSON
            msgs = ex["messages"]
            tools = ex.get("tools")
            norm = json.dumps(msgs, sort_keys=True, ensure_ascii=False)
            if tools:
                norm += json.dumps(tools, sort_keys=True, ensure_ascii=False)
            h = hashlib.sha256(norm.encode()).hexdigest()
            if h in hashes:
                dups += 1
                print("  %s line %d dup of line %d (id=%s)" % (cat, i, hashes[h], ex["id"]))
            else:
                hashes[h] = i
    print("%s: %d total, %d unique, %d duplicates" % (cat, i, len(hashes), dups))
    print()
