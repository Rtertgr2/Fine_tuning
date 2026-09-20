"""Category rules for `plan` examples (P1–P4).

P1  final answer is pure JSON (no fences, no greeting)  err
P2  has status, target_version, critique, final_plan   err
P3  status is one of the allowed values                err
P4  target_version looks like a version                warn

Allowed status values (decision, docs/decisions.md): APPROVED | NEEDS_REVISION
"""

from __future__ import annotations

import re

from backend.validators.base import (
    Example,
    ValidationContext,
    Violation,
    parse_final_json,
    v,
)

REQUIRED_FIELDS = ("status", "target_version", "critique", "final_plan")
_VERSION_RE = re.compile(r"^v\d+(\.\d+){1,2}$")


class P1_PureJson:
    code = "P1"
    level = "err"
    categories = frozenset({"plan"})
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        _obj, error = parse_final_json(ex)
        if error:
            return [v(self.code, self.level, f"final answer is not pure JSON: {error}")]
        return []


class P2_Fields:
    code = "P2"
    level = "err"
    categories = frozenset({"plan"})
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        obj, error = parse_final_json(ex)
        if error or obj is None:
            return []  # P1 reports the parse failure
        missing = [f for f in REQUIRED_FIELDS if f not in obj]
        if missing:
            return [v(self.code, self.level, f"missing field(s): {', '.join(missing)}")]
        return []


class P3_StatusValue:
    code = "P3"
    level = "err"
    categories = frozenset({"plan"})
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        obj, error = parse_final_json(ex)
        if error or obj is None or "status" not in obj:
            return []
        status = obj["status"]
        if status not in ctx.plan_status_values:
            return [
                v(
                    self.code,
                    self.level,
                    f"status {status!r} not in {list(ctx.plan_status_values)}",
                )
            ]
        return []


class P4_VersionFormat:
    code = "P4"
    level = "warn"
    categories = frozenset({"plan"})
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        obj, error = parse_final_json(ex)
        if error or obj is None or "target_version" not in obj:
            return []
        version = obj["target_version"]
        if not isinstance(version, str) or not _VERSION_RE.match(version):
            return [
                v(
                    self.code,
                    self.level,
                    f"target_version {version!r} does not match v<major>.<minor>[.<patch>]",
                )
            ]
        return []
