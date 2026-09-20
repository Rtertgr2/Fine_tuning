"""Model adapters: everything model-specific lives behind this interface.

The workbench stays model-agnostic: validators, API and pipelines talk to a
ModelAdapter, which encapsulates:

- the chat template (vendored jinja file — an exact copy from the model repo)
- how tool calls / tool results are embedded in message text
- a real tokenizer for counting (downloaded on demand)

Canonical message storage (agreed in plan 00, verified in T1.0):
  assistant content carries tool calls as embedded tags, exactly as the model
  emits them:  <tool_call>\\n{"name": ..., "arguments": {...}}\\n</tool_call>
  tool results are raw text in a {"role": "tool"} message; the template adds
  the <tool_response> wrapper itself.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    json_text: str  # exact JSON text found between the tags
    block_text: str  # full block including the tags


@dataclass(frozen=True)
class ParseIssue:
    kind: str  # unclosed_tag | stray_close_tag | invalid_json | bad_shape | stray_text
    message: str


@dataclass(frozen=True)
class ParsedAssistant:
    calls: list[ToolCall]
    issues: list[ParseIssue]
    stray_text: str  # non-whitespace text outside any tool_call block


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def content_hash(messages: Sequence[dict], tools: Sequence[dict] | None) -> str:
    """Stable hash of an example's meaningful content (for duplicate checks)."""
    payload = json.dumps(
        {"messages": list(messages), "tools": list(tools) if tools else None},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def shingles(text: str, n: int = 5) -> set[str]:
    """Character n-grams for near-duplicate detection (works for Thai too)."""
    t = normalize_text(text)
    if len(t) <= n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Adapter base class
# ---------------------------------------------------------------------------


class ModelAdapter:
    """Base class; concrete adapters set templates + metadata."""

    name: str = ""
    display_name: str = ""
    hf_repo: str = ""
    bos_token: str = ""
    eos_token: str = ""

    tool_call_open: str = "<tool_call>"
    tool_call_close: str = "</tool_call>"
    tool_response_open: str = "<tool_response>"
    tool_response_close: str = "</tool_response>"

    default_template_file: str = ""
    tool_use_template_file: str | None = None

    # -- templates ----------------------------------------------------------

    def template_text(self, kind: str) -> str:
        """kind: 'default' | 'tool_use'"""
        filename = {
            "default": self.default_template_file,
            "tool_use": self.tool_use_template_file,
        }.get(kind)
        if not filename:
            raise ValueError(f"adapter {self.name!r} has no {kind!r} template")
        return (TEMPLATES_DIR / filename).read_text(encoding="utf-8")

    def pick_template(self, tools: Sequence[dict] | None) -> str:
        """tool_use template when tools are present, else the default one."""
        if tools and self.tool_use_template_file:
            return "tool_use"
        return "default"

    def template_hash(self, kind: str) -> str:
        return hashlib.sha256(self.template_text(kind).encode("utf-8")).hexdigest()

    # -- tool call text -----------------------------------------------------

    def render_tool_call(self, name: str, arguments: dict[str, Any]) -> str:
        """Canonical embedded form — must equal what the template emits for
        the same call in structured form (verified by test_render_parity)."""
        payload = json.dumps(
            {"name": name, "arguments": arguments}, ensure_ascii=False
        )
        return f"{self.tool_call_open}\n{payload}\n{self.tool_call_close}"

    def parse_tool_calls(self, text: str) -> ParsedAssistant:
        """Parse embedded tool-call tags out of assistant content."""
        calls: list[ToolCall] = []
        issues: list[ParseIssue] = []
        spans: list[tuple[int, int]] = []

        idx = 0
        while True:
            open_at = text.find(self.tool_call_open, idx)
            if open_at == -1:
                break
            close_at = text.find(self.tool_call_close, open_at + len(self.tool_call_open))
            if close_at == -1:
                issues.append(
                    ParseIssue(
                        "unclosed_tag",
                        f"<{self.tool_call_open.strip('<>')}> opened at offset {open_at} "
                        "has no closing tag",
                    )
                )
                spans.append((open_at, len(text)))
                break
            inner = text[open_at + len(self.tool_call_open) : close_at].strip()
            block_end = close_at + len(self.tool_call_close)
            spans.append((open_at, block_end))
            try:
                obj = json.loads(inner)
            except json.JSONDecodeError as exc:
                issues.append(
                    ParseIssue("invalid_json", f"tool_call JSON is invalid: {exc}")
                )
            else:
                if not isinstance(obj, dict):
                    issues.append(
                        ParseIssue("bad_shape", "tool_call must be a JSON object")
                    )
                else:
                    name = obj.get("name")
                    args = obj.get("arguments")
                    if not isinstance(name, str) or not name:
                        issues.append(
                            ParseIssue("bad_shape", "tool_call needs a non-empty 'name'")
                        )
                    elif not isinstance(args, dict):
                        issues.append(
                            ParseIssue(
                                "bad_shape",
                                "tool_call 'arguments' must be a JSON object",
                            )
                        )
                    else:
                        calls.append(
                            ToolCall(
                                name=name,
                                arguments=args,
                                json_text=inner,
                                block_text=text[open_at:block_end],
                            )
                        )
            idx = block_end

        # stray close tags (more closes than opens)
        opens = text.count(self.tool_call_open)
        closes = text.count(self.tool_call_close)
        if closes > len(spans) or (opens == 0 and closes > 0):
            issues.append(
                ParseIssue(
                    "stray_close_tag",
                    f"found {closes} closing tag(s) but only {opens} opening tag(s)",
                )
            )

        # text outside any block
        remainder = text
        for start, end in reversed(spans):
            remainder = remainder[:start] + remainder[end:]
        stray = remainder.strip()

        return ParsedAssistant(calls=calls, issues=issues, stray_text=stray)

    # -- rendering ----------------------------------------------------------

    def render_conversation(
        self,
        messages: Sequence[dict],
        tools: Sequence[dict] | None = None,
        add_generation_prompt: bool = False,
        template_kind: str | None = None,
    ) -> str:
        """Render a conversation with the vendored template (local, offline).

        Uses transformers' own jinja compiler so the tojson filter behaviour is
        identical to what the model's tokenizer does at training time.
        """
        from backend.adapters.render import render_jinja

        kind = template_kind or self.pick_template(tools)
        return render_jinja(
            template_text=self.template_text(kind),
            messages=list(messages),
            tools=list(tools) if tools else [],
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            add_generation_prompt=add_generation_prompt,
        )

    def to_hf_messages(self, messages: Sequence[dict]) -> list[dict]:
        """Convert embedded-tag storage into HF structured form (tool_calls
        field). Used by parity tests and any structured pipeline."""
        out: list[dict] = []
        for msg in messages:
            if msg.get("role") == "assistant":
                parsed = self.parse_tool_calls(msg.get("content") or "")
                if parsed.calls and not parsed.stray_text:
                    out.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": c.name,
                                        "arguments": c.arguments,
                                    },
                                }
                                for c in parsed.calls
                            ],
                        }
                    )
                    continue
            out.append(dict(msg))
        return out

    # -- tokens -------------------------------------------------------------

    def count_tokens(self, text: str) -> int | None:
        """Real tokenizer count; None when the tokenizer is unavailable."""
        from backend.adapters.tokenizers import count_tokens

        return count_tokens(self, text)

    def tokenizer_available(self) -> bool:
        from backend.adapters.tokenizers import is_available

        return is_available(self)
