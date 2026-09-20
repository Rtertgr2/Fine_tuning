"""Validator entry points — one place that lists all rules.

`validate_example` runs the per-example rules (C*, T*, L1-3, P*, S1-5);
`validate_dataset` runs the dataset-scope rules (C6 peers handled inside
examples as well, plus L4, S6, S7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from backend.adapters.registry import get_adapter
from backend.app import config
from backend.validators import common, loop_rules, plan_rules, sec_rules, tool_rules
from backend.validators.base import Example, Level, ValidationContext, Violation

# ---------------------------------------------------------------------------
# Rule inventory (order matters for report readability)
# ---------------------------------------------------------------------------

COMMON_RULES = [
    common.C1_Roles(),
    common.C2_NonEmpty(),
    common.C3_HasAssistant(),
    common.C4_MaxTokens(),
    common.C5_NoSecrets(),
    common.C6_Duplicates(),
]

CATEGORY_RULES: dict[str, list] = {
    "tool": [
        tool_rules.T1_ToolCallSyntax(),
        tool_rules.T2_KnownTool(),
        tool_rules.T3_Arguments(),
        tool_rules.T4_CallFollowedByResult(),
        tool_rules.T5_CallCount(),
        tool_rules.T6_FullFileWrite(),
    ],
    "loop": [
        tool_rules.T1_ToolCallSyntax(),
        tool_rules.T2_KnownTool(),
        tool_rules.T3_Arguments(),
        tool_rules.T4_CallFollowedByResult(),
        loop_rules.L1_HasFailure(),
        loop_rules.L2_NextCallDiffers(),
        loop_rules.L3_NoByteIdenticalRepeat(),
    ],
    "plan": [
        plan_rules.P1_PureJson(),
        plan_rules.P2_Fields(),
        plan_rules.P3_StatusValue(),
        plan_rules.P4_VersionFormat(),
    ],
    "sec": [
        sec_rules.S1_PureJson(),
        sec_rules.S2_StatusValue(),
        sec_rules.S3_IssuesShape(),
        sec_rules.S4_LinesExist(),
        sec_rules.S5_PassHasNoIssues(),
    ],
}

DATASET_RULES = [
    loop_rules.L4_GiveupShare(),
    sec_rules.S6_PassShare(),
    sec_rules.S7_TypeCoverage(),
]


@dataclass
class Report:
    ok: bool
    violations: list[Violation] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[Violation]:
        return [x for x in self.violations if x.level == "err"]

    @property
    def warnings(self) -> list[Violation]:
        return [x for x in self.violations if x.level == "warn"]


def make_context(
    peers: Sequence[Example] = (),
    adapter_name: str | None = None,
    max_seq_len: int | None = None,
) -> ValidationContext:
    adapter = get_adapter(adapter_name)
    return ValidationContext(
        adapter=adapter,
        max_seq_len=max_seq_len if max_seq_len is not None else config.MAX_SEQ_LEN,
        peers=list(peers),
        tokenizer_available=adapter.tokenizer_available(),
        near_dup_threshold=config.NEAR_DUP_THRESHOLD,
        l4_giveup_ratio=config.L4_GIVEUP_RATIO,
        s6_pass_window=config.S6_PASS_WINDOW,
        s7_min_per_type=config.S7_MIN_PER_TYPE,
        plan_status_values=config.PLAN_STATUS_VALUES,
    )


def report_to_dict(report: Report) -> dict:
    return {
        "ok": report.ok,
        "violations": [
            {
                "code": x.code,
                "level": x.level,
                "message": x.message,
                "location": x.location,
            }
            for x in report.violations
        ],
        "skipped": report.skipped,
    }


def rules_for(category: str) -> list:
    return COMMON_RULES + CATEGORY_RULES.get(category, [])


def all_rules_for(category: str) -> list:
    """Example-scope rules plus the dataset-scope rules that observe this category."""
    dataset_side = [
        r for r in DATASET_RULES if getattr(r, "categories", None) and category in r.categories
    ]
    return rules_for(category) + dataset_side


def validate_example(ex: Example, ctx: ValidationContext) -> Report:
    violations: list[Violation] = []
    skipped: list[str] = []
    for rule in rules_for(ex.category):
        if rule.code == "C4" and not ctx.tokenizer_available:
            skipped.append("C4")
            continue
        violations.extend(rule.check(ex, ctx))
    violations.sort(key=lambda x: (x.code, x.location or ""))
    return Report(ok=not any(x.level == "err" for x in violations), violations=violations, skipped=skipped)


def validate_dataset(examples: Sequence[Example], ctx: ValidationContext) -> Report:
    violations: list[Violation] = []
    skipped: list[str] = []
    for ex in examples:
        sub = validate_example(ex, ctx)
        violations.extend(sub.violations)
        skipped.extend(sub.skipped)
    for rule in DATASET_RULES:
        violations.extend(rule.check(examples, ctx))
    violations.sort(key=lambda x: (x.code, x.location or ""))
    return Report(
        ok=not any(x.level == "err" for x in violations),
        violations=violations,
        skipped=sorted(set(skipped)),
    )