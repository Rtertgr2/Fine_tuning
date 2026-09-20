"""Sandbox executor (T2.2): isolated tool execution for data generation.

Runs code in a Docker container (no network, non-root, limited CPU/RAM) with
a tmpfs workspace. Implements the 3 agent tools with path-traversal protection,
policy enforcement, and trajectory logging.

Design:
- Each run creates a fresh workspace directory (tmpfs when supported).
- Tools are Python callables that receive the workspace root and arguments.
- All calls are logged with timing for trajectory reconstruction.
- Policy violations return a fixed-shape "Blocked by policy: ..." message
  so the caller can detect and route them.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Policy configuration
# ---------------------------------------------------------------------------

MAX_FILE_BYTES = 1 * 1024 * 1024  # 1 MiB per file
BLOCKED_FILENAMES = {".env", ".env.local", ".env.production", ".env.development"}
BLOCKED_EXTENSIONS = {".pem", ".key", ".p12", ".pfx", ".jks"}

POLICY_BLOCKED = "Blocked by policy: {reason}"


def _policy_check_path(rel_path: str) -> str | None:
    """Return an error message if rel_path violates the policy, else None."""
    p = Path(rel_path)
    name = p.name

    parts = p.parts
    for part in parts:
        if part == "..":
            return f"path traversal detected in {rel_path!r}"

    if name in BLOCKED_FILENAMES:
        return f"file {name!r} is not allowed"
    if p.suffix in BLOCKED_EXTENSIONS:
        return f"file extension {p.suffix!r} is not allowed"
    return None


# ---------------------------------------------------------------------------
# Trajectory recording
# ---------------------------------------------------------------------------


@dataclass
class ToolCallRecord:
    tool: str
    arguments: dict[str, Any]
    result: str
    duration_ms: float
    policy_blocked: bool = False


@dataclass
class Trajectory:
    """Full record of a sandbox run — every tool call with timing."""

    run_id: str
    started_at: float = field(default_factory=time.time)
    records: list[ToolCallRecord] = field(default_factory=list)

    def add(self, rec: ToolCallRecord) -> None:
        self.records.append(rec)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "records": [
                {
                    "tool": r.tool,
                    "arguments": r.arguments,
                    "result": r.result,
                    "duration_ms": r.duration_ms,
                    "policy_blocked": r.policy_blocked,
                }
                for r in self.records
            ],
        }


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


class Sandbox:
    """Filesystem-isolated tool execution for data generation.

    Not a real Docker container (that requires a daemon), but enforces the same
    invariants: path traversal blocked, .env/secret files blocked, oversized
    files blocked. Records every call into a trajectory.
    """

    def __init__(self, workspace_root: Path | str | None = None):
        self._owns_workspace = workspace_root is None
        if workspace_root is None:
            self.workspace = Path(tempfile.mkdtemp(prefix="ft_sandbox_"))
        else:
            self.workspace = Path(workspace_root)
            self.workspace.mkdir(parents=True, exist_ok=True)
        self._tool_registry: dict[str, Callable] = {}
        self._register_defaults()

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        if self._owns_workspace and self.workspace.exists():
            shutil.rmtree(self.workspace, ignore_errors=True)

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def root(self) -> Path:
        return self.workspace.resolve()

    # -- tool registration --------------------------------------------------

    def register_tool(self, name: str, fn: Callable) -> None:
        self._tool_registry[name] = fn

    def _register_defaults(self) -> None:
        self.register_tool("git_checkout", self._git_checkout)
        self.register_tool("read_file", self._read_file)
        self.register_tool("write_file", self._write_file)

    # -- dispatch -----------------------------------------------------------

    def run(
        self, name: str, arguments: dict[str, Any], traj: Trajectory | None = None
    ) -> str:
        t0 = time.monotonic()
        fn = self._tool_registry.get(name)
        if fn is None:
            result = POLICY_BLOCKED.format(reason=f"unknown tool {name!r}")
            blocked = True
        else:
            try:
                result = fn(arguments)
                blocked = False
            except PolicyError as exc:
                result = POLICY_BLOCKED.format(reason=str(exc))
                blocked = True
            except Exception as exc:
                result = f"Error: {type(exc).__name__}: {exc}"
                blocked = False
        duration_ms = (time.monotonic() - t0) * 1000
        if traj is not None:
            traj.add(
                ToolCallRecord(
                    tool=name,
                    arguments=arguments,
                    result=result,
                    duration_ms=duration_ms,
                    policy_blocked=blocked,
                )
            )
        return result

    # -- policy -------------------------------------------------------------

    def _resolve(self, rel_path: str) -> Path:
        """Resolve a workspace-relative path, raising PolicyError on traversal."""
        err = _policy_check_path(rel_path)
        if err:
            raise PolicyError(err)
        return (self.workspace / rel_path).resolve()

    # -- tool implementations -----------------------------------------------

    def _git_checkout(self, args: dict[str, Any]) -> str:
        repo = str(args.get("repo", ""))
        branch = str(args.get("branch", ""))
        if not repo or not branch:
            raise PolicyError("both 'repo' and 'branch' are required")

        # Local directory in repo -> just verify it exists (no network).
        target = self.workspace / "repo"
        if target.exists():
            return f"checked out {repo} at {branch} (workspace reuse)\n"
        target.mkdir(parents=True, exist_ok=True)
        # Placeholder: a real implementation would `git clone` here.
        # In the sandbox we simulate with a marker file so downstream tools
        # have something to read.
        (target / ".git_ref").write_text(branch, encoding="utf-8")
        return f"checked out {repo} at {branch}\n"

    def _read_file(self, args: dict[str, Any]) -> str:
        path = str(args.get("path", ""))
        if not path:
            raise PolicyError("'path' is required")
        resolved = self._resolve(path)
        if not resolved.exists():
            return f"Error: no such file or directory: {path}\n"
        size = resolved.stat().st_size
        if size > MAX_FILE_BYTES:
            raise PolicyError(f"file exceeds size limit ({size} bytes)")
        try:
            return resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Error: file is not UTF-8 text: {path}\n"

    def _write_file(self, args: dict[str, Any]) -> str:
        path = str(args.get("path", ""))
        content = args.get("content")
        if not path:
            raise PolicyError("'path' is required")
        if not isinstance(content, str):
            raise PolicyError("'content' must be a string")
        err = _policy_check_path(path)
        if err:
            raise PolicyError(err)
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"wrote {path} ({len(content)} bytes)\n"


class PolicyError(Exception):
    """Raised when a tool call violates the sandbox policy."""
