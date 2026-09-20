#!/usr/bin/env python3
"""Print unique sec examples to understand what we have."""
import json

seen = set()
with open("data/gold/sec.jsonl") as f:
    for i, line in enumerate(f, 1):
        ex = json.loads(line)
        msgs = ex["messages"]
        # Get status from assistant JSON
        for m in msgs:
            if m["role"] == "assistant":
                try:
                    obj = json.loads(str(m["content"]))
                    status = obj.get("status", "?")
                    n_issues = len(obj.get("issues", []))
                    issue_types = [iss.get("type", "?") for iss in obj.get("issues", [])]
                    print("line %d: %s %s (%d issues: %s)" % (i, ex["id"], status, n_issues, issue_types))
                except:
                    print("line %d: %s (parse error)" % (i, ex["id"]))
                break
