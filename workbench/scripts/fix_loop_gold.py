#!/usr/bin/env python3
"""Fix loop duplicates: lines 22-24 dup of line 21. Add unique loop examples."""
import json, hashlib
from pathlib import Path

GOLD = Path("data/gold")

LOOP_SYSTEM = "You are a coding agent with access to tools: git_checkout, read_file, write_file. Complete the user's task by calling tools. Every tool call must be wrapped in <tool_call> tags with JSON containing name and arguments. Tool results appear as <tool_response> messages. Write full file contents, never diffs or partial fragments."

TOOLS = [
    {"type": "function", "function": {"name": "git_checkout", "description": "Check out a repository.", "parameters": {"type": "object", "properties": {"repo": {"type": "string"}, "branch": {"type": "string"}}, "required": ["repo", "branch"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
]

def tc(name, args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "\n</tool_call>"

def tr(content):
    return "<tool_response>\n" + content + "\n</tool_response>"

# New unique loop examples (failure + recovery + giveup patterns)
new_loops = [
    # File not found -> change path -> succeed
    {
        "id": "ex_gold_loop_new_001",
        "category": "loop",
        "messages": [
            {"role": "system", "content": LOOP_SYSTEM},
            {"role": "user", "content": "Read the config file and update the database URL."},
            {"role": "assistant", "content": tc("read_file", {"path": "config/database.yaml"})},
            {"role": "tool", "content": tr("Error: No such file or directory: config/database.yaml")},
            {"role": "assistant", "content": tc("read_file", {"path": "config/db.yml"})},
            {"role": "tool", "content": tr("database:\n  url: sqlite:///old.db\n")},
            {"role": "assistant", "content": tc("write_file", {"path": "config/db.yml", "content": "database:\n  url: postgresql://localhost/mydb\n"})},
            {"role": "tool", "content": tr("File written successfully")},
            {"role": "assistant", "content": "Done. Found the config at config/db.yml and updated the database URL."},
        ],
        "tools": TOOLS,
        "meta": {"source": "manual", "group": "gold/loop/not-found-change-path", "status": "approved"},
    },
    # Syntax error in file -> read and fix -> succeed
    {
        "id": "ex_gold_loop_new_002",
        "category": "loop",
        "messages": [
            {"role": "system", "content": LOOP_SYSTEM},
            {"role": "user", "content": "Run the test suite and fix any failures."},
            {"role": "assistant", "content": tc("read_file", {"path": "tests/test_main.py"})},
            {"role": "tool", "content": tr("def test_add():\n    assert add(1, 2 = 3\n")},
            {"role": "assistant", "content": tc("write_file", {"path": "tests/test_main.py", "content": "def test_add():\n    assert add(1, 2) == 3\n\ndef test_sub():\n    assert sub(5, 3) == 2\n"})},
            {"role": "tool", "content": tr("File written successfully")},
            {"role": "assistant", "content": "Fixed the syntax error (missing closing parenthesis) and added a new test."},
        ],
        "tools": TOOLS,
        "meta": {"source": "manual", "group": "gold/loop/syntax-fix", "status": "approved"},
    },
    # Permission denied -> read different file -> giveup
    {
        "id": "ex_gold_loop_new_003",
        "category": "loop",
        "messages": [
            {"role": "system", "content": LOOP_SYSTEM},
            {"role": "user", "content": "Read the SSH private key to verify its format."},
            {"role": "assistant", "content": tc("read_file", {"path": "/root/.ssh/id_rsa"})},
            {"role": "tool", "content": tr("Permission denied: /root/.ssh/id_rsa")},
            {"role": "assistant", "content": tc("read_file", {"path": "~/.ssh/id_rsa"})},
            {"role": "tool", "content": tr("Permission denied: ~/.ssh/id_rsa")},
            {"role": "assistant", "content": "I cannot complete this task. The SSH private key file is inaccessible due to permission restrictions. Reporting the failure: insufficient permissions to read the private key."},
        ],
        "tools": TOOLS,
        "meta": {"source": "manual", "group": "gold/loop/permission-giveup", "status": "approved"},
    },
]

# Read existing, remove duplicates (lines 22-24 which are 0-indexed 21-23)
existing = []
with open(GOLD / "loop.jsonl") as f:
    for line in f:
        existing.append(json.loads(line))

# Remove the known duplicates (ex_gold_loop_022, 023, 024)
dup_ids = {"ex_gold_loop_022", "ex_gold_loop_023", "ex_gold_loop_024"}
filtered = [ex for ex in existing if ex["id"] not in dup_ids]

# Add new
for ex in new_loops:
    filtered.append(ex)

# Deduplicate by hash
seen = set()
unique = []
for ex in filtered:
    norm = json.dumps(ex["messages"], sort_keys=True, ensure_ascii=False)
    if ex.get("tools"):
        norm += json.dumps(ex["tools"], sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(norm.encode()).hexdigest()
    if h not in seen:
        seen.add(h)
        unique.append(ex)

with open(GOLD / "loop.jsonl", "w") as f:
    for ex in unique:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

# Count giveup
from backend.validators.loop_rules import ends_with_giveup, is_failed_result
from backend.adapters.registry import get_adapter
adapter = get_adapter("hermes2pro-llama3-8b")
from backend.validators.base import Example
giveups = 0
for ex in unique:
    fw = Example(id=ex["id"], category="loop", messages=ex["messages"], tools=ex.get("tools"))
    if ends_with_giveup(fw, adapter):
        giveups += 1

print("loop.jsonl: %d unique examples" % len(unique))
print("  Giveup: %d (%.0f%%)" % (giveups, giveups/len(unique)*100))
