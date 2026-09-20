"""Category rules for `sec` examples (S1–S7) — security review output.

S1  final answer is pure JSON                        err
S2  status is PASS or REJECT                         err
S3  REJECT has issues[] with file/line/type/severity/fix  err
S4  every referenced line exists in the given diff   err
S5  PASS has no issues                               warn
S6  dataset: PASS share near 50%                     warn (dataset level)
S7  dataset: vulnerability-type coverage >= N each    warn (dataset level)
"""

from __future__ import annotations

import re
from typing import Sequence

from backend.validators.base import (
    Example,
    ValidationContext,
    Violation,
    parse_final_json,
    v,
)

ISSUE_FIELDS = ("file", "line", "type", "severity", "fix")
SEVERITIES = {"critical", "high", "medium", "low"}

_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

# S7 required type families (config-extendable; see docs/decisions.md)
DEFAULT_REQUIRED_TYPES = ["sql_injection", "resource_leak", "insecure_deserialization", "owasp"]


def extract_diff_line_numbers(text: str) -> set[int] | None:
    """New-file line numbers covered by unified-diff hunks in `text`.

    Counts added (+) and context (space) lines; removed (-) lines consume no
    new-file number. Returns None when no hunk header is found.
    """
    lines = text.splitlines()
    current: int | None = None
    valid: set[int] = set()
    saw_hunk = False
    for line in lines:
        m = _HUNK_RE.match(line)
        if m:
            saw_hunk = True
            current = int(m.group(1))
            continue
        if current is None:
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            valid.add(current)
            current += 1
        elif line.startswith("-"):
            continue
        elif line.startswith(" ") or line == "":
            valid.add(current)
            current += 1
    return valid if saw_hunk else None


def _diff_text(ex: Example) -> str:
    return "\n".join(
        str(m.get("content") or "") for m in ex.messages if m.get("role") in ("user", "tool")
    )


class S1_PureJson:
    code = "S1"
    level = "err"
    categories = frozenset({"sec"})
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        _obj, error = parse_final_json(ex)
        if error:
            return [v(self.code, self.level, f"final answer is not pure JSON: {error}")]
        return []


class S2_StatusValue:
    code = "S2"
    level = "err"
    categories = frozenset({"sec"})
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        obj, error = parse_final_json(ex)
        if error or obj is None or "status" not in obj:
            return []
        if obj["status"] not in ("PASS", "REJECT"):
            return [v(self.code, self.level, f"status {obj['status']!r} must be PASS or REJECT")]
        return []


class S3_IssuesShape:
    code = "S3"
    level = "err"
    categories = frozenset({"sec"})
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        obj, error = parse_final_json(ex)
        if error or obj is None or obj.get("status") != "REJECT":
            return []
        out: list[Violation] = []
        issues = obj.get("issues")
        if not isinstance(issues, list) or not issues:
            return [v(self.code, self.level, "REJECT must carry a non-empty issues list")]
        for k, issue in enumerate(issues):
            if not isinstance(issue, dict):
                out.append(v(self.code, self.level, f"issues[{k}] must be an object"))
                continue
            missing = [f for f in ISSUE_FIELDS if f not in issue]
            if missing:
                out.append(
                    v(self.code, self.level, f"issues[{k}] missing field(s): {', '.join(missing)}")
                )
                continue
            if not isinstance(issue["line"], int) or isinstance(issue["line"], bool):
                out.append(v(self.code, self.level, f"issues[{k}].line must be an integer"))
            if not str(issue["file"]).strip():
                out.append(v(self.code, self.level, f"issues[{k}].file must not be empty"))
            if not str(issue["type"]).strip():
                out.append(v(self.code, self.level, f"issues[{k}].type must not be empty"))
            if not str(issue["fix"]).strip():
                out.append(v(self.code, self.level, f"issues[{k}].fix must not be empty"))
            if str(issue["severity"]).lower() not in SEVERITIES:
                out.append(
                    v(
                        self.code,
                        self.level,
                        f"issues[{k}].severity {issue['severity']!r} not in {sorted(SEVERITIES)}",
                    )
                )
        return out


class S4_LinesExist:
    code = "S4"
    level = "err"
    categories = frozenset({"sec"})
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        obj, error = parse_final_json(ex)
        if error or obj is None or obj.get("status") != "REJECT":
            return []
        issues = obj.get("issues")
        if not isinstance(issues, list):
            return []
        valid = extract_diff_line_numbers(_diff_text(ex))
        if valid is None:
            return [
                v(
                    self.code,
                    self.level,
                    "no unified diff found in the example — cannot verify issue line numbers",
                )
            ]
        out = []
        for k, issue in enumerate(issues):
            if not isinstance(issue, dict):
                continue
            line = issue.get("line")
            if isinstance(line, int) and not isinstance(line, bool) and line not in valid:
                out.append(
                    v(
                        self.code,
                        self.level,
                        f"issues[{k}].line {line} is not a line of the provided diff",
                    )
                )
        return out


class S5_PassHasNoIssues:
    code = "S5"
    level = "warn"
    categories = frozenset({"sec"})
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        obj, error = parse_final_json(ex)
        if error or obj is None or obj.get("status") != "PASS":
            return []
        issues = obj.get("issues")
        if isinstance(issues, list) and len(issues) > 0:
            return [v(self.code, self.level, "PASS with a non-empty issues list")]
        return []


class S6_PassShare:
    code = "S6"
    level = "warn"
    scope = "dataset"
    categories = frozenset({"sec"})

    def check(self, examples: Sequence[Example], ctx: ValidationContext) -> list[Violation]:
        sec = [e for e in examples if e.category == "sec"]
        if not sec:
            return []
        pass_count = 0
        counted = 0
        for e in sec:
            obj, error = parse_final_json(e)
            if error or obj is None:
                continue
            status = obj.get("status")
            if status in ("PASS", "REJECT"):
                counted += 1
                if status == "PASS":
                    pass_count += 1
        if not counted:
            return []
        ratio = pass_count / counted
        low, high = ctx.s6_pass_window
        if not (low <= ratio <= high):
            return [
                v(
                    self.code,
                    self.level,
                    f"PASS share is {ratio:.0%} ({pass_count}/{counted}); want {low:.0%}–{high:.0%}",
                )
            ]
        return []


class S7_TypeCoverage:
    code = "S7"
    level = "warn"
    scope = "dataset"
    categories = frozenset({"sec"})

    def check(self, examples: Sequence[Example], ctx: ValidationContext) -> list[Violation]:
        sec = [e for e in examples if e.category == "sec"]
        if not sec:
            return []
        required = getattr(ctx, "s7_required_types", None) or DEFAULT_REQUIRED_TYPES
        min_each = ctx.s7_min_per_type
        counts: dict[str, int] = {t: 0 for t in required}
        for e in sec:
            obj, error = parse_final_json(e)
            if error or obj is None or obj.get("status") != "REJECT":
                continue
            types = {
                str(issue.get("type", "")).lower()
                for issue in obj.get("issues", [])
                if isinstance(issue, dict)
            }
            for t in required:
                if any(vt == t or vt.startswith(t + "_") or (t == "owasp" and vt.startswith("a0")) for vt in types):
                    counts[t] += 1
        short = {t: c for t, c in counts.items() if c < min_each}
        if short:
            detail = ", ".join(f"{t}={c}" for t, c in sorted(short.items()))
            return [
                v(
                    self.code,
                    self.level,
                    f"vulnerability-type coverage below {min_each} examples: {detail}",
                )
            ]
        return []