"""Shared fixtures."""

from __future__ import annotations

import pytest

from backend.app import config


@pytest.fixture()
def tmp_workbench(tmp_path, monkeypatch):
    """Isolated DB + dataset dir for a test."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "examples.db")
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")
    from backend.app.db import init_db

    init_db(tmp_path / "examples.db")
    return tmp_path


@pytest.fixture()
def conn(tmp_workbench):
    from backend.app.db import connect

    c = connect(tmp_workbench / "examples.db")
    yield c
    c.close()


# ---------------------------------------------------------------------------
# example builders
# ---------------------------------------------------------------------------


def msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def tool_call_block(name: str, arguments: dict) -> str:
    import json

    return f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)}\n</tool_call>'


def read_call(path: str = "a.py") -> str:
    return tool_call_block("read_file", {"path": path})


def write_call(path: str, content: str) -> str:
    return tool_call_block("write_file", {"path": path, "content": content})


@pytest.fixture()
def tool_example():
    """A valid tool-category example: git_checkout → read → write, 3 calls."""
    from backend.validators.base import Example

    messages = [
        msg("system", "You are a coding agent. Use the tools to complete the task."),
        msg("user", "checkout repo demo, อ่าน a.py แล้วเขียน hello.py ให้พิมพ์ hello world"),
        msg(
            "assistant",
            tool_call_block("git_checkout", {"repo": "demo", "branch": "main"}),
        ),
        msg("tool", "checked out demo at main\n"),
        msg("assistant", read_call("a.py")),
        msg("tool", "print('hi')\n"),
        msg("assistant", write_call("hello.py", "print('hello world')\n")),
        msg("tool", "wrote hello.py (20 bytes)\n"),
        msg("assistant", "เสร็จแล้วครับ เขียน hello.py เรียบร้อย"),
    ]
    return Example(
        id="ex_test_tool",
        category="tool",
        messages=messages,
        tools=None,
        group_id="repo/test",
    )


@pytest.fixture()
def loop_example():
    """A valid loop example: failed read, recovery, give-up report."""
    from backend.validators.base import Example

    messages = [
        msg("system", "You are a coding agent."),
        msg("user", "ช่วยอ่านไฟล์ b.py ให้หน่อย"),
        msg("assistant", read_call("b.py")),
        msg("tool", "Error: no such file or directory: b.py"),
        msg("assistant", read_call("src/b.py")),
        msg("tool", "print('found it')\n"),
        msg("assistant", "เจอไฟล์แล้วครับ เนื้อหาคือ print('found it')"),
    ]
    return Example(id="ex_test_loop", category="loop", messages=messages, group_id="repo/test")


@pytest.fixture()
def plan_example():
    from backend.validators.base import Example

    messages = [
        msg("system", "You are a plan reviewer. Answer with JSON only."),
        msg("user", "ตรวจแผนนี้: ..."),
        msg(
            "assistant",
            '{"status": "APPROVED", "target_version": "v0.2.0", '
            '"critique": "แผนครอบคลุมดี", "final_plan": "ดำเนินการตามแผน"}',
        ),
    ]
    return Example(id="ex_test_plan", category="plan", messages=messages, group_id="plan/1")


@pytest.fixture()
def sec_example():
    from backend.validators.base import Example

    diff = (
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1,5 +1,6 @@\n"
        " import sqlite3\n"
        "-def get_user(name):\n"
        "+def get_user(name):\n"
        "+    query = \"SELECT * FROM users WHERE name = '\" + name + \"'\"\n"
        "     cur.execute(query)\n"
    )
    messages = [
        msg("system", "You are a security reviewer. Answer with JSON only."),
        msg("user", f"ตรวจ diff นี้:\n{diff}"),
        msg(
            "assistant",
            '{"status": "REJECT", "issues": [{"file": "app.py", "line": 3, '
            '"type": "sql_injection", "severity": "critical", '
            '"fix": "ใช้ parameterized query"}]}',
        ),
    ]
    return Example(id="ex_test_sec", category="sec", messages=messages, group_id="repo/sec-1")
