"""Adapter parsing / hashing behaviour."""

from __future__ import annotations

from backend.adapters.base import content_hash, jaccard, shingles
from backend.adapters.registry import get_adapter

ADAPTER = get_adapter()


def test_parse_single_call():
    block = ADAPTER.render_tool_call("read_file", {"path": "a.py"})
    parsed = ADAPTER.parse_tool_calls(block)
    assert not parsed.issues
    assert parsed.stray_text == ""
    assert len(parsed.calls) == 1
    assert parsed.calls[0].name == "read_file"
    assert parsed.calls[0].arguments == {"path": "a.py"}


def test_parse_multiple_calls():
    text = ADAPTER.render_tool_call("read_file", {"path": "a.py"}) + "\n" + ADAPTER.render_tool_call(
        "git_checkout", {"repo": "r", "branch": "main"}
    )
    parsed = ADAPTER.parse_tool_calls(text)
    assert [c.name for c in parsed.calls] == ["read_file", "git_checkout"]


def test_parse_unclosed_tag():
    parsed = ADAPTER.parse_tool_calls("<tool_call>\n{\"name\": \"read_file\"}")
    assert any(i.kind == "unclosed_tag" for i in parsed.issues)
    assert not parsed.calls


def test_parse_invalid_json():
    parsed = ADAPTER.parse_tool_calls("<tool_call>\nnot json\n</tool_call>")
    assert any(i.kind == "invalid_json" for i in parsed.issues)


def test_parse_bad_shape():
    parsed = ADAPTER.parse_tool_calls('<tool_call>\n{"name": "read_file", "arguments": "x"}\n</tool_call>')
    assert any(i.kind == "bad_shape" for i in parsed.issues)


def test_parse_stray_text():
    text = "ผมจะอ่านไฟล์ก่อน\n" + ADAPTER.render_tool_call("read_file", {"path": "a.py"})
    parsed = ADAPTER.parse_tool_calls(text)
    assert parsed.stray_text == "ผมจะอ่านไฟล์ก่อน"
    assert len(parsed.calls) == 1


def test_parse_stray_close_tag():
    parsed = ADAPTER.parse_tool_calls("</tool_call>")
    assert any(i.kind == "stray_close_tag" for i in parsed.issues)


def test_content_hash_stable_and_whitespace_sensitive():
    msgs_a = [{"role": "user", "content": "hi"}]
    msgs_b = [{"role": "user", "content": "hi"}]
    msgs_c = [{"role": "user", "content": "hi "}]
    assert content_hash(msgs_a, None) == content_hash(msgs_b, None)
    assert content_hash(msgs_a, None) != content_hash(msgs_c, None)


def test_shingles_and_jaccard():
    a = shingles("อ่านไฟล์ a.py แล้วแก้ไขให้เรียบร้อย")
    b = shingles("อ่านไฟล์ a.py แล้วแก้ไขให้เรียบร้อย")
    c = shingles("ตรวจสอบความปลอดภัยของโค้ด")
    assert jaccard(a, b) == 1.0
    assert jaccard(a, c) < 0.3


def test_to_hf_messages_roundtrip():
    block = ADAPTER.render_tool_call("read_file", {"path": "a.py"})
    msgs = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": block},
        {"role": "tool", "content": "r"},
        {"role": "assistant", "content": "done"},
    ]
    hf = ADAPTER.to_hf_messages(msgs)
    assert hf[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert hf[1]["tool_calls"][0]["function"]["arguments"] == {"path": "a.py"}
    # tool and plain messages unchanged
    assert hf[2] == {"role": "tool", "content": "r"}
    assert hf[3] == {"role": "assistant", "content": "done"}
