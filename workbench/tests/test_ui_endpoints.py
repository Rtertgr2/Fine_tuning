"""UI support endpoints (T1.5): /render draft preview, /tools registry, /ui static."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from backend.app.main import app

from .conftest import msg, tool_call_block


def client() -> TestClient:
    return TestClient(app)


def _draft_plan() -> dict:
    return {
        "category": "plan",
        "messages": [
            msg("system", "You are a plan reviewer. Answer with JSON only."),
            msg("user", "ตรวจแผนนี้: ..."),
            msg(
                "assistant",
                json.dumps(
                    {
                        "status": "APPROVED",
                        "target_version": "v0.2.0",
                        "critique": "แผนครอบคลุมดี",
                        "final_plan": "ดำเนินการตามแผน",
                    },
                    ensure_ascii=False,
                ),
            ),
        ],
        "group_id": "plan/1",
    }


def _draft_tool() -> dict:
    return {
        "category": "tool",
        "messages": [
            msg("system", "You are a coding agent."),
            msg("user", "อ่าน a.py"),
            msg("assistant", tool_call_block("read_file", {"path": "a.py"})),
            msg("tool", "print('hi')\n"),
        ],
        "tools": None,
    }


class TestRenderEndpoint:
    def test_render_matches_adapter_template(self, tmp_workbench):
        with client() as c:
            res = c.post("/render", json=_draft_plan())
        assert res.status_code == 200
        body = res.json()
        # Hermes ChatML (with BOS): system turn wrapped <|im_start|> ... <|im_end|>
        assert body["rendered"].startswith("<|begin_of_text|><|im_start|>system\n")
        assert body["rendered"].rstrip().endswith("<|im_end|>")
        assert body["template_kind"] == "default"
        assert body["tools_auto_injected"] is False
        assert body["max_seq_len"] >= 1

    def test_render_tool_draft_auto_injects_tools(self, tmp_workbench):
        with client() as c:
            res = c.post("/render", json=_draft_tool())
        body = res.json()
        assert res.status_code == 200
        assert body["template_kind"] == "tool_use"
        # the tool definitions must be inside the rendered system preamble
        assert '"read_file"' in body["rendered"]
        assert body["tools_auto_injected"] is True

    def test_render_token_count_consistent_with_availability(self, tmp_workbench):
        with client() as c:
            body = c.post("/render", json=_draft_plan()).json()
        if body["tokenizer_available"]:
            assert isinstance(body["token_count"], int) and body["token_count"] > 0
        else:
            assert body["token_count"] is None


class TestToolsEndpoint:
    def test_tools_registry(self, tmp_workbench):
        with client() as c:
            res = c.get("/tools")
        assert res.status_code == 200
        body = res.json()
        assert body["names"] == ["git_checkout", "read_file", "write_file"]
        assert len(body["schemas"]) == 3
        names = {s["function"]["name"] for s in body["schemas"]}
        assert names == set(body["names"])


class TestUiStatic:
    def test_ui_index_served(self, tmp_workbench):
        with client() as c:
            res = c.get("/ui/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "Dataset Studio" in res.text

    def test_ui_index_at_root_path(self, tmp_workbench):
        with client() as c:
            res = c.get("/ui")
        assert res.status_code in (200, 307)  # redirect to /ui/ or direct hit
