"""Second adapter (Qwen2.5) — proves the model-agnostic layer works."""

from __future__ import annotations

from backend.adapters.registry import get_adapter

QWEN = get_adapter("qwen2.5-7b-instruct")
HERMES = get_adapter("hermes2pro-llama3-8b")


def test_registry_lists_both():
    from backend.adapters.registry import list_adapters

    names = {a["name"] for a in list_adapters()}
    assert {"hermes2pro-llama3-8b", "qwen2.5-7b-instruct"} <= names


def test_qwen_renders_tool_use():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "p"}},
                    "required": ["path"],
                },
            },
        }
    ]
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
    ]
    text = QWEN.render_conversation(msgs, tools=tools)
    assert "# Tools" in text
    assert "<tool_call>" in text  # instructions mention the tag


def test_qwen_parses_same_tag_format():
    block = QWEN.render_tool_call("read_file", {"path": "a.py"})
    parsed = QWEN.parse_tool_calls(block)
    assert parsed.calls[0].name == "read_file"
    assert not parsed.issues


def test_each_adapter_uses_its_own_template():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    hermes_text = HERMES.render_conversation(msgs)
    qwen_text = QWEN.render_conversation(msgs)
    assert hermes_text.startswith("<|begin_of_text|>")
    assert not qwen_text.startswith("<|begin_of_text|>")
    assert hermes_text != qwen_text