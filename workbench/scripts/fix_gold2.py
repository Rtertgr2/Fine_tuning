#!/usr/bin/env python3
"""Fix remaining gold errors:
- sec S4: strip leading whitespace from diff blocks in user messages
- tool C5: redact password references in write_file content
"""
import json, re
from pathlib import Path

GOLD = Path("data/gold")

# ── SEC fix: strip diff indentation ──────────────────────────
def fix_sec_whitespace():
    path = GOLD / "sec.jsonl"
    lines = path.read_text().splitlines()
    fixed = []
    for line in lines:
        ex = json.loads(line)
        for m in ex["messages"]:
            if m["role"] == "user":
                content = str(m.get("content", "") or "")
                # Strip leading spaces from diff lines (@@, +, -, space-prefix)
                new_lines = []
                for dl in content.split("\n"):
                    stripped = dl.lstrip()
                    if stripped.startswith("@@") or stripped.startswith("+") or stripped.startswith("-") or stripped.startswith("Review this diff"):
                        new_lines.append(stripped)
                    else:
                        new_lines.append(dl)
                m["content"] = "\n".join(new_lines)
        fixed.append(json.dumps(ex, ensure_ascii=False))
    path.write_text("\n".join(fixed) + "\n")
    print("sec: stripped diff whitespace in %d examples" % len(fixed))

# ── TOOL fix: redact passwords ───────────────────────────────
def fix_tool_passwords():
    path = GOLD / "tool.jsonl"
    lines = path.read_text().splitlines()
    fixed = []
    for line in lines:
        ex = json.loads(line)
        for m in ex["messages"]:
            if m["role"] == "assistant":
                content = str(m.get("content", "") or "")
                if "DB_PASSWORD" in content or "password:" in content.lower():
                    # Redact password values
                    content = content.replace("${DB_PASSWORD}", "[REDACTED]")
                    content = re.sub(r'(?i)password:\s*\S+', 'password: [REDACTED]', content)
                    m["content"] = content
        fixed.append(json.dumps(ex, ensure_ascii=False))
    path.write_text("\n".join(fixed) + "\n")
    print("tool: redacted passwords in affected examples")

if __name__ == "__main__":
    fix_sec_whitespace()
    fix_tool_passwords()
