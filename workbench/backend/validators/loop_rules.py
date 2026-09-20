"""Category rules for `loop` examples (L1–L4) — failure and recovery.

L1  has at least one failed tool result                 warn
L2  the call after a failure differs from the failed one err
L3  no byte-identical repeated call after failure       err
L4  dataset: >=15% end with a give-up report            warn (dataset level)
"""

from __future__ import annotations

from typing import Any, Sequence

from backend.validators.base import (
    Example,
    ValidationContext,
    Violation,
    v,
)

_FAILURE_MARKERS = [
    "error",
    "not found",
    "no such file",
    "traceback",
    "failed",
    "failure",
    "permission denied",
    "denied",
    "cannot",
    "can't",
    "unable",
    "invalid",
    "ไม่พบ",
    "ผิดพลาด",
    "ล้มเหลว",
    "ไม่สามารถ",
]

_GIVEUP_MARKERS = [
    "cannot complete",
    "can't complete",
    "could not complete",
    "unable to complete",
    "cannot proceed",
    "can't proceed",
    "unable to proceed",
    "cannot continue",
    "can't continue",
    "stopping here",
    "stopped",
    "giving up",
    "report the failure",
    "reporting the failure",
    "ไม่สามารถดำเนินการ",
    "ไม่สามารถทำต่อ",
    "ไม่สามารถดำเนินการต่อ",
    "ทำไม่ได้",
    "ต้องหยุด",
    "ขอรายงาน",
    "รายงานว่าล้มเหลว",
]


def is_failed_result(content: str) -> bool:
    text = (content or "").strip().lower()
    if not text:
        return False
    if text.startswith("error") or text.startswith("failed"):
        return True
    if text.startswith("{") and '"error"' in text:
        return True
    return any(marker in text for marker in _FAILURE_MARKERS)


def _call_result_pairs(ex: Example, adapter: Any) -> list[tuple[Any, dict, int, int]]:
    """Pair each tool_call with its result message: (call, result_msg, call_idx, result_idx)."""
    pairs: list[tuple[Any, dict, int, int]] = []
    msgs = ex.messages
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        calls = adapter.parse_tool_calls(m.get("content") or "").calls
        if not calls:
            continue
        j, k = i + 1, 0
        while j < len(msgs) and msgs[j].get("role") == "tool":
            if k < len(calls):
                pairs.append((calls[k], msgs[j], i, j))
            k += 1
            j += 1
    return pairs


def ends_with_giveup(ex: Example, adapter: Any) -> bool:
    """Last assistant message reports it could not finish, with no further calls."""
    last = None
    for m in reversed(ex.messages):
        if m.get("role") == "assistant":
            last = m
            break
    if last is None:
        return False
    content = last.get("content") or ""
    if adapter.parse_tool_calls(content).calls:
        return False
    lowered = content.lower()
    return any(marker in lowered for marker in _GIVEUP_MARKERS)


class L1_HasFailure:
    code = "L1"
    level = "warn"
    categories = frozenset({"loop"})
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        pairs = _call_result_pairs(ex, ctx.adapter)
        if not pairs:
            return []  # T1/T4 report missing calls
        if not any(is_failed_result(msg.get("content") or "") for _, msg, _, _ in pairs):
            return [
                v(
                    self.code,
                    self.level,
                    "loop example has no failed tool result — it does not exercise recovery",
                )
            ]
        return []


class L2_NextCallDiffers:
    code = "L2"
    level = "err"
    categories = frozenset({"loop"})
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        pairs = _call_result_pairs(ex, ctx.adapter)
        out = []
        for idx, (call, result, _ci, ri) in enumerate(pairs):
            if not is_failed_result(result.get("content") or ""):
                continue
            following = pairs[idx + 1 :]
            if not following:
                continue  # nothing after — L1-style stop is fine
            next_call = following[0][0]
            if next_call.name == call.name and next_call.arguments == call.arguments:
                out.append(
                    v(
                        self.code,
                        self.level,
                        f"call after failure repeats the failed call {call.name!r} unchanged",
                        f"messages[{ri}]",
                    )
                )
        return out


class L3_NoByteIdenticalRepeat:
    code = "L3"
    level = "err"
    categories = frozenset({"loop"})
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        pairs = _call_result_pairs(ex, ctx.adapter)
        out = []
        for idx, (call, result, _ci, ri) in enumerate(pairs):
            if not is_failed_result(result.get("content") or ""):
                continue
            for later_call, _r, _li, _ri in pairs[idx + 1 :]:
                if later_call.block_text == call.block_text:
                    out.append(
                        v(
                            self.code,
                            self.level,
                            f"byte-identical tool_call repeated after failure ({call.name!r})",
                            f"messages[{ri}]",
                        )
                    )
                    break
        return out


class L4_GiveupShare:
    code = "L4"
    level = "warn"
    scope = "dataset"
    categories = frozenset({"loop"})

    def check(self, examples: Sequence[Example], ctx: ValidationContext) -> list[Violation]:
        loops = [e for e in examples if e.category == "loop"]
        if not loops:
            return []
        giveups = sum(1 for e in loops if ends_with_giveup(e, ctx.adapter))
        ratio = giveups / len(loops)
        if ratio < ctx.l4_giveup_ratio:
            return [
                v(
                    self.code,
                    self.level,
                    f"only {ratio:.0%} of loop examples end with a give-up report "
                    f"(want >= {ctx.l4_giveup_ratio:.0%}: {giveups}/{len(loops)})",
                )
            ]
        return []
