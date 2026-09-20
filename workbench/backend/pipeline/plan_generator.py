"""Plan-category generator (T2.6): plan critique examples with injected flaws.

Takes plan prompts and injects specific flaws (missing rollback, wrong version,
ambiguous steps), then generates the expected critique based on the injected
flaws (ground truth comes from the injection, not the model's guess).
"""

from __future__ import annotations

import json
import random
from typing import Any, Sequence

from backend.pipeline.generator import BaseGenerator

SYSTEM_PROMPT = """You are a plan reviewer. Review the following plan and respond with JSON only:
{"status": "APPROVED" or "NEEDS_REVISION", "target_version": "vX.Y.Z", "critique": "your analysis", "final_plan": "the final plan text"}"""

# Flaw types for injection
PLAN_FLAWS = {
    "missing_rollback": {
        "description": "plan does not describe rollback steps",
        "critique_addition": "แผนไม่มีขั้นตอน rollback เมื่อเกิดปัญหา",
    },
    "wrong_version": {
        "description": "version does not follow semver or is inconsistent",
        "critique_addition": "เวอร์ชันไม่สอดคล้องกับ semantic versioning",
    },
    "ambiguous_steps": {
        "description": "steps are vague and not actionable",
        "critique_addition": "ขั้นตอนไม่ชัดเจน ไม่สามารถดำเนินการได้เป็นรูปธรรม",
    },
    "missing_verification": {
        "description": "plan has no verification step after changes",
        "critique_addition": "แผนไม่มีขั้นตอนตรวจสอบผลลัพธ์หลังเปลี่ยนแปลง",
    },
    "missing_dependencies": {
        "description": "plan does not mention required dependencies",
        "critique_addition": "ไม่ได้ระบุ dependencies ที่จำเป็น",
    },
}


class PlanGenerator(BaseGenerator):
    """Generate plan-category examples with injected flaws and known critiques."""

    category = "plan"
    source = "generated"

    def __init__(self, db_path=None, budget_usd=None, flaw_types=None, clean_ratio=0.3):
        super().__init__(db_path, budget_usd)
        self._flaw_types = flaw_types or list(PLAN_FLAWS.keys())
        self._clean_ratio = clean_ratio  # fraction of examples without flaws

    def _generate_candidates(
        self,
        seeds: Sequence[dict[str, Any]],
        trajectory_factory: Any | None = None,
    ) -> list[dict[str, Any]]:
        candidates = []
        for seed in seeds:
            # Decide if this should be clean (APPROVED) or flawed (NEEDS_REVISION)
            is_clean = random.random() < self._clean_ratio
            flaw_type = "unknown"

            if is_clean:
                traj = self._generate_clean_plan(seed)
            else:
                flaw_type = random.choice(self._flaw_types)
                traj = self._generate_flawed_plan(seed, flaw_type)

            if traj is not None:
                candidates.append({
                    "category": "plan",
                    "messages": traj["messages"],
                    "tools": None,
                    "group_id": seed.get("group_id", f"gen/plan/{flaw_type if not is_clean else 'clean'}"),
                    "meta": {
                        "injected_flaw": None if is_clean else flaw_type,
                        "status": traj["status"],
                    },
                })
        return candidates

    def _generate_clean_plan(self, seed: dict[str, Any]) -> dict[str, Any] | None:
        """Generate a plan that passes review (APPROVED)."""
        plan_text = seed.get("plan", self._default_plan())
        version = seed.get("version", "v1.0.0")

        critique = "แผนครอบคลุมดี มีขั้นตอนชัดเจน มีการ rollback และตรวจสอบผลลัพธ์"
        final_plan = plan_text

        response = {
            "status": "APPROVED",
            "target_version": version,
            "critique": critique,
            "final_plan": final_plan,
        }

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"ตรวจแผนนี้:\n{plan_text}\n\nเวอร์ชันเป้าหมาาย: {version}"},
            {"role": "assistant", "content": json.dumps(response, ensure_ascii=False)},
        ]

        return {"messages": messages, "status": "APPROVED"}

    def _generate_flawed_plan(self, seed: dict[str, Any], flaw_type: str) -> dict[str, Any] | None:
        """Generate a plan with an injected flaw and the expected critique."""
        flaw = PLAN_FLAWS[flaw_type]
        plan_text = seed.get("plan", self._default_flawed_plan(flaw_type))
        version = seed.get("version", "v1.0.0")

        critique = flaw["critique_addition"]
        # Add a constructive suggestion
        if flaw_type == "missing_rollback":
            suggestion = " ควรเพิ่มขั้นตอน rollback"
        elif flaw_type == "wrong_version":
            suggestion = " ควรแก้ไขเวอร์ชันให้ถูกต้องตาม semver"
        elif flaw_type == "ambiguous_steps":
            suggestion = " ควรระบุขั้นตอนให้ชัดเจนและปฏิบัติการได้จริง"
        elif flaw_type == "missing_verification":
            suggestion = " ควรเพิ่มขั้นตอน verify หลังเปลี่ยนแปลง"
        else:
            suggestion = " ควรระบุ dependencies ที่จำเป็น"

        critique += suggestion

        response = {
            "status": "NEEDS_REVISION",
            "target_version": version,
            "critique": critique,
            "final_plan": plan_text + f"\n\n# TODO: {suggestion.strip()}",
        }

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"ตรวจแผนนี้:\n{plan_text}\n\nเวอร์ชันเป้าหมาย: {version}"},
            {"role": "assistant", "content": json.dumps(response, ensure_ascii=False)},
        ]

        return {"messages": messages, "status": "NEEDS_REVISION"}

    def _default_plan(self) -> str:
        return (
            "## แผนการพัฒนา\n"
            "1. วิเคราะห์ความต้องการ\n"
            "2. ออกแบบโครงสร้าง\n"
            "3. พัฒนาฟีเจอร์\n"
            "4. ทดสอบและ verify\n"
            "5. rollback หากพบปัญหา"
        )

    def _default_flawed_plan(self, flaw_type: str) -> str:
        if flaw_type == "missing_rollback":
            return (
                "## แผนการพัฒนา\n"
                "1. วิเคราะห์ความต้องการ\n"
                "2. ออกแบบโครงสร้าง\n"
                "3. พัฒนาฟีเจอร์\n"
                "4. deploy ทันที"
            )
        elif flaw_type == "wrong_version":
            return (
                "## แผนการพัฒนา\n"
                "1. วิเคราะห์ความต้องการ\n"
                "2. พัฒนาฟีเจอร์\n"
                "3. deploy เวอร์ชัน 1.0"
            )
        elif flaw_type == "ambiguous_steps":
            return (
                "## แผนการพัฒนา\n"
                "1. ทำงานตามที่วางไว้\n"
                "2. ดูผล\n"
                "3. ส่งมอบ"
            )
        else:
            return (
                "## แผนการพัฒนา\n"
                "1. วิเคราะห์ความต้องการ\n"
                "2. พัฒนาฟีเจอร์\n"
                "3. deploy"
            )
