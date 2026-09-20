"""Real git checkout tool (T2.2) — wraps sandbox with actual git operations.

For data generation, this performs a real `git clone` / `git checkout` into
the sandbox workspace. Network access is required; the caller is responsible
for ensuring the sandbox has network access for this specific operation.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def git_checkout(workspace: Path, args: dict[str, Any]) -> str:
    """Clone/checkout a repo into the workspace.

    Returns a status message. Raises RuntimeError on git failure.
    """
    repo = str(args.get("repo", ""))
    branch = str(args.get("branch", ""))
    if not repo or not branch:
        raise ValueError("both 'repo' and 'branch' are required")

    target = workspace / "repo"
    if (target / ".git").exists():
        # Already cloned — just checkout the ref.
        _run_git(["fetch", "--all"], cwd=target)
        _run_git(["checkout", branch], cwd=target)
        return f"checked out {repo} at {branch}\n"

    target.parent.mkdir(parents=True, exist_ok=True)
    _run_git(["clone", "--branch", branch, "--", repo, str(target)], cwd=workspace)
    return f"cloned {repo} at {branch}\n"


def _run_git(args: list[str], cwd: Path) -> str:
    cmd = ["git"] + args
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout
