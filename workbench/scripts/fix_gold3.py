#!/usr/bin/env python3
"""Fix sec S4 issue lines + tool C5 password content."""
import json, re
from pathlib import Path

GOLD = Path("data/gold")

# ── SEC: fix issue.line numbers ──────────────────────────────
def fix_sec():
    path = GOLD / "sec.jsonl"
    lines = path.read_text().splitlines()
    fixed = []
    for line in lines:
        ex = json.loads(line)
        for m in ex["messages"]:
            if m["role"] != "assistant":
                continue
            content = str(m.get("content", "") or "")
            try:
                obj = json.loads(content)
            except:
                continue
            if "issues" not in obj:
                continue
            changed = False
            for iss in obj["issues"]:
                ln = iss.get("line")
                # Fix known bad lines: reduce by 1 (off-by-one from diff parsing)
                if ex["id"] in ("ex_gold_sec_023", "ex_gold_sec_024", "ex_gold_sec_025") and ln == 5:
                    iss["line"] = 4
                    changed = True
                elif ex["id"] in ("ex_gold_sec_031", "ex_gold_sec_032") and ln == 4:
                    iss["line"] = 3
                    changed = True
            if changed:
                m["content"] = json.dumps(obj, ensure_ascii=False)
        fixed.append(json.dumps(ex, ensure_ascii=False))
    path.write_text("\n".join(fixed) + "\n")
    print("sec: fixed line numbers in affected examples")

# ── TOOL: remove password fields entirely ────────────────────
def fix_tool():
    path = GOLD / "tool.jsonl"
    lines = path.read_text().splitlines()
    fixed = []
    for line in lines:
        ex = json.loads(line)
        for m in ex["messages"]:
            if m["role"] == "assistant":
                content = str(m.get("content", "") or "")
                if "write_file" in content and "password" in content.lower():
                    # Parse tool_call, redact password in arguments.content
                    try:
                        # Find JSON inside tool_call tags
                        match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)
                        if match:
                            tc = json.loads(match.group(1))
                            file_content = tc.get("arguments", {}).get("content", "")
                            # Remove password lines
                            new_lines = []
                            for fl in file_content.split("\n"):
                                if re.match(r'\s*password\s*:', fl, re.IGNORECASE):
                                    new_lines.append(fl.split(":")[0] + ': ""')
                                else:
                                    new_lines.append(fl)
                            tc["arguments"]["content"] = "\n".join(new_lines)
                            new_tc = json.dumps(tc, ensure_ascii=False)
                            content = content.replace(match.group(1), new_tc)
                            m["content"] = content
                    except:
                        pass
        fixed.append(json.dumps(ex, ensure_ascii=False))
    path.write_text("\n".join(fixed) + "\n")
    print("tool: cleared password values in affected examples")

if __name__ == "__main__":
    fix_sec()
    fix_tool()
