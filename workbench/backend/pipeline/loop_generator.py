"""Loop-category generator (T2.5): anti-loop and self-correction examples.

Takes successful tool trajectories and injects errors at a specific step k,
then generates the correct recovery behavior.
"""

from __future__ import annotations

import json
import random
from typing import Any, Sequence

from backend.adapters.registry import get_adapter
from backend.pipeline.generator import BaseGenerator
from backend.tools.registry import tool_schemas

SYSTEM_PROMPT = """You are a coding agent. Use the available tools to complete the task.
When a tool call fails, analyze the error and try a different approach.
If you cannot recover after 2 attempts, report the failure with a clear reason."""

# Error types for injection
ERROR_INJECTIONS = {
    "file_not_found": {
        "tool": "read_file",
        "modify_args": lambda args: {**args, "path": "nonexistent/" + args.get("path", "file.txt")},
        "expected_error": "Error: no such file or directory",
    },
    "syntax_error": {
        "tool": "write_file",
        "modify_args": lambda args: {**args, "content": args.get("content", "") + "def broken(\n"},
        "expected_error": "syntax error",
    },
    "blocked_by_policy": {
        "tool": "read_file",
        "modify_args": lambda args: {"path": ".env"},
        "expected_error": "Blocked by policy",
    },
    "branch_not_found": {
        "tool": "git_checkout",
        "modify_args": lambda args: {**args, "branch": "nonexistent-branch-xyz"},
        "expected_error": "error",
    },
}


class LoopGenerator(BaseGenerator):
    """Generate loop-category examples by injecting errors into trajectories."""

    category = "loop"
    source = "generated"

    def __init__(self, db_path=None, budget_usd=None, error_types=None):
        super().__init__(db_path, budget_usd)
        self._error_types = error_types or list(ERROR_INJECTIONS.keys())

    def _generate_candidates(
        self,
        seeds: Sequence[dict[str, Any]],
        trajectory_factory: Any | None = None,
    ) -> list[dict[str, Any]]:
        candidates = []
        for seed in seeds:
            # Pick an error type to inject
            error_type = random.choice(self._error_types)
            injection = ERROR_INJECTIONS[error_type]

            traj = self._inject_and_recover(seed, injection)
            if traj is not None:
                candidates.append({
                    "category": "loop",
                    "messages": traj["messages"],
                    "tools": tool_schemas(),
                    "group_id": seed.get("group_id", f"gen/loop/{error_type}"),
                    "meta": {
                        "injected_error": error_type,
                        "injected_at_step": traj.get("injected_at", 0),
                        "recovery_type": traj.get("recovery_type", "unknown"),
                    },
                })
        return candidates

    def _inject_and_recover(
        self,
        seed: dict[str, Any],
        injection: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Generate a trajectory with an injected error and correct recovery.

        The recovery follows the loop-category rules:
        - L2: The next call must differ from the failed call.
        - L3: No byte-identical retry.
        - L1: There must be a visible failure in the trajectory.
        """
        from backend.tools.sandbox import Sandbox, Trajectory
        from backend.app import ids
        from backend.adapters.base import content_hash

        run_id = ids.new_ulid()
        traj = Trajectory(run_id=run_id)
        sandbox = Sandbox()

        task = seed.get("task", "")
        repo = seed.get("repo", "")
        branch = seed.get("branch", "main")

        try:
            # Step 1: git_checkout
            sandbox.run(
                "git_checkout", {"repo": repo, "branch": branch}, traj=traj
            )

            # Step 2: read the target file (will succeed)
            read_path = seed.get("target_file", "app.py")
            success_read = sandbox.run(
                "read_file", {"path": read_path}, traj=traj
            )

            # Step 3: Inject the error
            injected_args = injection["modify_args"]({"path": read_path, "content": "pass\n"})
            injected_result = sandbox.run(
                injection["tool"], injected_args, traj=traj
            )
            injected_at = len(traj.records) - 1

            # Step 4: Recovery based on error type
            recovery_type = "unknown"
            if injection["tool"] == "read_file" and "path" in injected_args:
                # Recovery: list directory or try alternate path
                # For sandbox, re-read the correct path
                recovery_result = sandbox.run(
                    "read_file", {"path": read_path}, traj=traj
                )
                recovery_type = "retry_with_correct_path"

            elif injection["tool"] == "write_file":
                # Recovery: read the file first to verify, then write full content
                verify_result = sandbox.run(
                    "read_file", {"path": read_path}, traj=traj
                )
                # Write valid content
                sandbox.run(
                    "write_file",
                    {"path": read_path + ".fixed", "content": "# Fixed content\npass\n"},
                    traj=traj,
                )
                recovery_type = "read_then_rewrite"

            elif injection["tool"] == "git_checkout":
                # Recovery: try main branch
                sandbox.run(
                    "git_checkout", {"repo": repo, "branch": "main"}, traj=traj
                )
                recovery_type = "fallback_branch"

            else:
                # Give up with report
                recovery_type = "give_up_report"

            # Build messages
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task},
            ]

            for rec in traj.records:
                call_text = self._render_tool_call(rec.tool, rec.arguments)
                messages.append({"role": "assistant", "content": call_text})
                messages.append({"role": "tool", "content": rec.result})

            # Final message depends on recovery type
            if recovery_type == "give_up_report":
                messages.append({
                    "role": "assistant",
                    "content": f"ไม่สามารถดำเนินการต่อได้ครับ พบข้อผิดพลาด: {injected_result.strip()}",
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": "แก้ไขข้อผิดพลาดและดำเนินการเสร็จสิ้นแล้วครับ",
                })

            return {
                "messages": messages,
                "injected_at": injected_at,
                "recovery_type": recovery_type,
            }
        finally:
            sandbox.close()

    def _render_tool_call(self, name: str, arguments: dict) -> str:
        adapter = get_adapter()
        return adapter.render_tool_call(name, arguments)
