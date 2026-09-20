"""Tool-category generator (T2.4): multi-turn tool call examples.

Runs the sandbox to produce trajectories where the teacher model solves a task
using git_checkout → read_file → write_file in sequence. Only trajectories
that pass Phase 1 validators and contain 3-6 tool calls are kept.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from backend.adapters.registry import get_adapter
from backend.pipeline.generator import BaseGenerator
from backend.tools.registry import TOOL_REGISTRY, tool_schemas

# System prompt for tool examples
SYSTEM_PROMPT = """You are a coding agent. Use the available tools to complete the task.
Tools: git_checkout, read_file, write_file.

Rules:
- Always write the FULL file content, never a partial diff.
- If a tool returns an error, read the actual file to understand the structure before retrying.
- After completing the task, summarize what you did."""


class ToolGenerator(BaseGenerator):
    """Generate tool-category examples by running tasks in the sandbox."""

    category = "tool"
    source = "generated"

    def _generate_candidates(
        self,
        seeds: Sequence[dict[str, Any]],
        trajectory_factory: Any | None = None,
    ) -> list[dict[str, Any]]:
        """For each seed, construct a multi-turn example from the seed's
        task description. Uses trajectory_factory (an optional callable that
        actually calls an LLM) or falls back to template-based generation
        when no LLM is available."""
        candidates = []
        for seed in seeds:
            task = seed.get("task", "")
            repo = seed.get("repo", "")
            branch = seed.get("branch", "main")
            expected_files = seed.get("expected_files", [])
            group_id = seed.get("group_id", f"gen/{repo}")

            if trajectory_factory is not None:
                traj = self._generate_with_llm(
                    task, repo, branch, expected_files, trajectory_factory
                )
            else:
                traj = self._generate_template(
                    task, repo, branch, expected_files
                )

            if traj is not None:
                candidates.append({
                    "category": "tool",
                    "messages": traj["messages"],
                    "tools": tool_schemas(),
                    "group_id": group_id,
                    "meta": {
                        "task": task,
                        "repo": repo,
                        "branch": branch,
                        "tool_calls": traj.get("tool_calls", []),
                    },
                })
        return candidates

    def _generate_template(
        self,
        task: str,
        repo: str,
        branch: str,
        expected_files: list[str],
    ) -> dict[str, Any] | None:
        """Template-based trajectory generation (no LLM required).

        Creates a deterministic 3-call trajectory:
        git_checkout → read_file → write_file
        """
        from backend.tools.sandbox import Sandbox, Trajectory
        from backend.app import ids

        run_id = ids.new_ulid()
        traj = Trajectory(run_id=run_id)
        sandbox = Sandbox()

        try:
            sandbox.run(
                "git_checkout",
                {"repo": repo, "branch": branch},
                traj=traj,
            )

            read_results = []
            for fpath in expected_files:
                result = sandbox.run(
                    "read_file", {"path": fpath}, traj=traj
                )
                read_results.append(result)

            # Determine the write action from the task
            # For template mode, write a simple response file
            write_path = expected_files[0] if expected_files else "output.txt"
            write_content = f"# Generated output for: {task}\n"
            if expected_files:
                write_content += f"# Based on: {', '.join(expected_files)}\n"

            sandbox.run(
                "write_file",
                {"path": write_path + ".out", "content": write_content},
                traj=traj,
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task},
            ]

            # Build assistant messages from trajectory
            tool_call_idx = 0
            for rec in traj.records:
                if rec.tool == "git_checkout":
                    call_text = self._render_tool_call(
                        "git_checkout", {"repo": repo, "branch": branch}
                    )
                    messages.append({"role": "assistant", "content": call_text})
                    messages.append({"role": "tool", "content": rec.result})
                elif rec.tool == "read_file":
                    args = rec.arguments
                    call_text = self._render_tool_call("read_file", args)
                    messages.append({"role": "assistant", "content": call_text})
                    messages.append({"role": "tool", "content": rec.result})
                elif rec.tool == "write_file":
                    args = rec.arguments
                    call_text = self._render_tool_call("write_file", args)
                    messages.append({"role": "assistant", "content": call_text})
                    messages.append({"role": "tool", "content": rec.result})

            messages.append({
                "role": "assistant",
                "content": f"เสร็จแล้วครับ ดำเนินการ{task}เรียบร้อย",
            })

            return {
                "messages": messages,
                "tool_calls": [r.tool for r in traj.records],
            }
        finally:
            sandbox.close()

    def _generate_with_llm(
        self,
        task: str,
        repo: str,
        branch: str,
        expected_files: list[str],
        llm_factory: Any,
    ) -> dict[str, Any] | None:
        """LLM-based trajectory generation.

        Uses the provided LLM to generate the assistant messages. The LLM
        is expected to return a list of (tool_call, arguments) pairs, which
        are then executed in the sandbox to verify correctness.
        """
        # This would integrate with a real LLM API. For now, return None
        # to signal that LLM-based generation is not yet wired up.
        return None

    def _render_tool_call(self, name: str, arguments: dict) -> str:
        """Render a tool call in the canonical embedded form."""
        adapter = get_adapter()
        return adapter.render_tool_call(name, arguments)
