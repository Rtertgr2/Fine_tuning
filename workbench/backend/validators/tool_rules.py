"""Category rules for `tool` examples (T1–T6).

T1  every <tool_call> is valid JSON with balanced tags   err
T2  tool name is in the registry                         err
T3  arguments complete and correctly typed               err
T4  every tool_call is followed by tool results          err
T5  call count is 3–6                                    warn
T6  write_file content is a full file, not a diff        err
"""

from __future__ import annotations

import re

from backend.tools.registry import check_arguments
from backend.validators.base import Example, ValidationContext, Violation, v


class T1_ToolCallSyntax:
    code = "T1"
    level = "err"
    categories = frozenset({"tool", "loop"})
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        out: list[Violation] = []
        found_any = False
        for i, m in enumerate(ex.messages):
            if m.get("role") != "assistant":
                continue
            parsed = ctx.adapter.parse_tool_calls(m.get("content") or "")
            if parsed.calls:
                found_any = True
            for issue in parsed.issues:
                out.append(v(self.code, self.level, issue.message, f"messages[{i}]"))
        if not found_any:
            out.append(v(self.code, self.level, "no tool_call found in a tool/loop example"))
        return out


class T2_KnownTool:
    code = "T2"
    level = "err"
    categories = frozenset({"tool", "loop"})
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        from backend.tools.registry import TOOL_REGISTRY

        out = []
        for i, m in enumerate(ex.messages):
            if m.get("role") != "assistant":
                continue
            for call in ctx.adapter.parse_tool_calls(m.get("content") or "").calls:
                if call.name not in TOOL_REGISTRY:
                    out.append(
                        v(
                            self.code,
                            self.level,
                            f"unknown tool {call.name!r}",
                            f"messages[{i}]",
                        )
                    )
        return out


class T3_Arguments:
    code = "T3"
    level = "err"
    categories = frozenset({"tool", "loop"})
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        out = []
        for i, m in enumerate(ex.messages):
            if m.get("role") != "assistant":
                continue
            for call in ctx.adapter.parse_tool_calls(m.get("content") or "").calls:
                for problem in check_arguments(call.name, call.arguments):
                    out.append(v(self.code, self.level, problem, f"messages[{i}]"))
        return out


class T4_CallFollowedByResult:
    code = "T4"
    level = "err"
    categories = frozenset({"tool", "loop"})
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        out = []
        msgs = ex.messages
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            calls = ctx.adapter.parse_tool_calls(m.get("content") or "").calls
            if not calls:
                continue
            n_tool_after = 0
            j = i + 1
            while j < len(msgs) and msgs[j].get("role") == "tool":
                n_tool_after += 1
                j += 1
            if n_tool_after < len(calls):
                out.append(
                    v(
                        self.code,
                        self.level,
                        f"{len(calls)} tool_call(s) but only {n_tool_after} tool result message(s) follow",
                        f"messages[{i}]",
                    )
                )
        return out


class T5_CallCount:
    code = "T5"
    level = "warn"
    categories = frozenset({"tool"})
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        total = 0
        for m in ex.messages:
            if m.get("role") == "assistant":
                total += len(ctx.adapter.parse_tool_calls(m.get("content") or "").calls)
        if total and not (3 <= total <= 6):
            return [v(self.code, self.level, f"tool example has {total} tool calls (want 3–6)")]
        return []


_DIFF_MARKERS = [
    (re.compile(r"(?m)^\s*@@ .* @@"), "unified-diff hunk header"),
    (re.compile(r"(?m)^\s*\+[^+~]"), "diff + line"),
    (re.compile(r"(?m)^\s*-[^-~]"), "diff - line"),
    (re.compile(r"<<<<<<< |>>>>>>> |=======\n"), "merge conflict marker"),
]
_PARTIAL_PHRASES = [
    "rest of the file",
    "remains unchanged",
    "remains the same",
    "unchanged",
    "same as before",
    "existing content",
    "ส่วนที่เหลือ",
    "เหมือนเดิม",
    "ไม่เปลี่ยนแปลง",
    "...",
]


class T6_FullFileWrite:
    code = "T6"
    level = "err"
    categories = frozenset({"tool", "loop"})
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        out = []
        for i, m in enumerate(ex.messages):
            if m.get("role") != "assistant":
                continue
            for call in ctx.adapter.parse_tool_calls(m.get("content") or "").calls:
                if call.name != "write_file":
                    continue
                content = call.arguments.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue  # T3 covers missing/empty
                for pattern, label in _DIFF_MARKERS:
                    if pattern.search(content):
                        out.append(
                            v(
                                self.code,
                                self.level,
                                f"write_file content looks like a diff ({label})",
                                f"messages[{i}]",
                            )
                        )
                        break
                lowered = content.lower()
                for phrase in _PARTIAL_PHRASES:
                    if phrase in lowered:
                        out.append(
                            v(
                                self.code,
                                self.level,
                                f"write_file content mentions a partial/unchanged region ({phrase!r})",
                                f"messages[{i}]",
                            )
                        )
                        break
        return out
