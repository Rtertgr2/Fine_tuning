"""Validator framework.

Rules are model-agnostic: they read examples and ask the active adapter to
parse tool calls / render / count tokens. Every rule declares:
  code      — stable id used in reports (C1, T2, L4, ...)
  level     — err (must not enter a dataset) or warn
  categories— which example categories it applies to (None = all)
  scope     — "example" or "dataset" (dataset rules run once per dataset)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence

Level = Literal["err", "warn"]


@dataclass(frozen=True)
class Violation:
    code: str
    level: Level
    message: str
    location: str | None = None


@dataclass
class Example:
    """Framework-level view of an example (decoupled from the API schema)."""

    id: str
    category: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    source: str = "manual"
    group_id: str | None = None
    status: str = "draft"
    content_hash: str = ""


@dataclass
class ValidationContext:
    adapter: Any  # backend.adapters.base.ModelAdapter
    max_seq_len: int = 8192
    peers: list[Example] = field(default_factory=list)
    tokenizer_available: bool = False
    near_dup_threshold: float = 0.85
    l4_giveup_ratio: float = 0.15
    s6_pass_window: tuple[float, float] = (0.35, 0.65)
    s7_min_per_type: int = 3
    plan_status_values: tuple[str, ...] = ("APPROVED", "NEEDS_REVISION")


class Rule(Protocol):
    code: str
    level: Level
    categories: frozenset[str] | None
    scope: str  # "example" | "dataset"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]: ...


class DatasetRule(Protocol):
    code: str
    level: Level
    scope: str  # "dataset"

    def check(self, examples: Sequence[Example], ctx: ValidationContext) -> list[Violation]: ...


# ---------------------------------------------------------------------------
# Shared helpers used by several rule modules
# ---------------------------------------------------------------------------


def assistant_messages(ex: Example) -> list[tuple[int, str]]:
    return [
        (i, m.get("content") or "")
        for i, m in enumerate(ex.messages)
        if m.get("role") == "assistant"
    ]


def final_assistant_content(ex: Example) -> str | None:
    for m in reversed(ex.messages):
        if m.get("role") == "assistant":
            return m.get("content") or ""
    return None


def parse_final_json(ex: Example) -> tuple[dict[str, Any] | None, str | None]:
    """Parse the last assistant message as a pure JSON object.

    Returns (obj, error). `error` describes why parsing failed.
    """
    content = final_assistant_content(ex)
    if content is None:
        return None, "no assistant message"
    text = content.strip()
    if text.startswith("```"):
        return None, "content starts with a code fence"
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(obj, dict):
        return None, "top-level JSON must be an object"
    return obj, None


def v(code: str, level: Level, message: str, location: str | None = None) -> Violation:
    return Violation(code=code, level=level, message=message, location=location)
