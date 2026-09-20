"""T1.0 verification — the tool-call text format must render byte-identically
through both paths:

  A) embedded tags in assistant content (our canonical storage)
  B) structured tool_calls field (what the model's own template renders)

If these ever diverge, training data and serving behaviour diverge.
"""

from __future__ import annotations

import pytest

from backend.adapters.registry import get_adapter
from backend.adapters.tokenizers import is_available

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "path"}},
                "required": ["path"],
            },
        },
    }
]


def _structured(call_specs, content="DONE", tools=TOOLS, adapter=None):
    adapter = adapter or get_adapter()
    calls = [
        {"type": "function", "function": {"name": n, "arguments": a}} for n, a in call_specs
    ]
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "ทำงานนี้ให้หน่อย"},
        {"role": "assistant", "content": None, "tool_calls": calls},
        {"role": "tool", "content": "print('hi')\n"},
        {"role": "assistant", "content": content},
    ]
    return adapter.render_conversation(msgs, tools=tools)


def _embedded(call_specs, content="DONE", tools=TOOLS, adapter=None):
    adapter = adapter or get_adapter()
    blocks = "\n".join(adapter.render_tool_call(n, a) for n, a in call_specs)
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "ทำงานนี้ให้หน่อย"},
        {"role": "assistant", "content": blocks},
        {"role": "tool", "content": "print('hi')\n"},
        {"role": "assistant", "content": content},
    ]
    return adapter.render_conversation(msgs, tools=tools)


def test_single_call_parity():
    specs = [("read_file", {"path": "a.py"})]
    assert _embedded(specs) == _structured(specs)


def test_multi_call_parity():
    specs = [
        ("read_file", {"path": "a.py"}),
        ("read_file", {"path": "b.py"}),
    ]
    assert _embedded(specs) == _structured(specs)


def test_unicode_arguments_parity():
    specs = [("read_file", {"path": "ไฟล์/ทดสอบ.py"})]
    assert _embedded(specs) == _structured(specs)


def test_nested_arguments_parity():
    specs = [
        (
            "write_file",
            {"path": "x.py", "content": "print('สวัสดี')\n# nested {json: [1,2]}"},
        )
    ]
    assert _embedded(specs) == _structured(specs)


def test_tool_response_rendering():
    adapter = get_adapter()
    text = _embedded([("read_file", {"path": "a.py"})])
    assert "<|im_start|>tool\n<tool_response>\nprint('hi')\n\n</tool_response>" in text


def test_rendered_text_contains_canonical_block():
    adapter = get_adapter()
    block = adapter.render_tool_call("read_file", {"path": "a.py"})
    text = _embedded([("read_file", {"path": "a.py"})])
    assert block in text


def test_default_template_used_without_tools():
    adapter = get_adapter()
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
    ]
    text = adapter.render_conversation(msgs, tools=None)
    assert "You are a function calling AI model" not in text
    assert text.startswith("<|begin_of_text|><|im_start|>system\nS<|im_end|>")


# ---------------------------------------------------------------------------
# The strongest check: our renderer vs the real tokenizer's own template.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not is_available(get_adapter()), reason="tokenizer not downloaded (offline)"
)
def test_render_matches_real_tokenizer_template():
    adapter = get_adapter()
    from backend.adapters.tokenizers import _load

    tok = _load(adapter)

    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "ทำงานนี้ให้หน่อย"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "read_file", "arguments": {"path": "a.py"}},
                }
            ],
        },
        {"role": "tool", "content": "print('hi')\n"},
        {"role": "assistant", "content": "เสร็จแล้ว"},
    ]

    ours = adapter.render_conversation(msgs, tools=TOOLS)
    theirs = tok.apply_chat_template(
        msgs, tools=TOOLS, tokenize=False, add_generation_prompt=False
    )
    assert ours == theirs
