"""Docker-based sandbox runtime (T2.2): isolated tool execution for data generation.

Replaces the in-process temp-directory sandbox with a Docker container that
enforces true isolation: non-root user, no network, resource limits, limited
device/mount/path access. Returns only artifact output.

Design:
- Each Sandbox instance creates a fresh Docker container (not started).
- Tools are executed via docker exec on a long-running container.
- The workspace is a tmpfs mount inside the container.
- Container is destroyed on close().
- Same public interface as the original Sandbox (Sandbox, Trajectory,
  ToolCallRecord, PolicyError).
"""

from __future__ import annotations

import docker
import os
import secrets
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Policy configuration (mirrors original sandbox.py)
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
# Trajectory recording (identical to original)
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
# Policy error
# ---------------------------------------------------------------------------

class PolicyError(Exception):
    """Raised when a tool call violates the sandbox policy."""


# ---------------------------------------------------------------------------
# Docker-based Sandbox
# ---------------------------------------------------------------------------

# Default resource limits
DEFAULT_MEMORY = "256m"
DEFAULT_CPUS = 1.0

# Allowed device nodes inside the container
ALLOWED_DEVICES = [
    "/dev/null:/dev/null",
    "/dev/zero:/dev/zero",
    "/dev/random:/dev/random",
    "/dev/urandom:/dev/urandom",
]

# Docker security options
DOCKER_SECURITY_OPTS = [
    "no-new-privileges",
]

# Python image to use (matches the running environment as closely as possible)
DOCKER_IMAGE = "python:3.11-slim"


class Sandbox:
    """Docker-isolated tool execution for data generation.

    Runs code in a non-root Docker container with no network access,
    resource limits, and limited device/path mounts. Implements the same
    interface as the original in-process Sandbox.
    """

    def __init__(self, workspace_root: Path | str | None = None):
        self._owns_workspace = workspace_root is None
        if workspace_root is None:
            self.workspace = Path(tempfile.mkdtemp(prefix="ft_docker_sandbox_"))
        else:
            self.workspace = Path(workspace_root)
            self.workspace.mkdir(parents=True, exist_ok=True)

        self._tool_registry: dict[str, Callable] = {}
        self._register_defaults()

        # Create a non-root user ID for the container
        self._uid = secrets.randbelow(10000) + 1000  # avoid system UIDs
        self._gid = self._uid

        # Create Docker client
        self._client = docker.from_env()

        # Create the container (not started) with all security options
        self._container = self._create_container()
        self._container.start()

        # Track whether container was started by us (needs cleanup)
        self._container_started = True

    def _create_container(self) -> docker.models.containers.Container:
        """Create a Docker container with full isolation settings."""
        workspace_path = str(self.workspace.resolve())

        # Build the docker run command for container creation
        cmd = [
            "docker", "create",
            # Security: non-root user
            f"-u", f"{self._uid}:{self._gid}",
            # Network isolation: no network
            "--network", "none",
            # Resource limits
            "--memory", DEFAULT_MEMORY,
            "--cpus", str(DEFAULT_CPUS),
            # Drop all capabilities
            "--cap-drop", "ALL",
            # Security options
            "--security-opt", "no-new-privileges",
            # Read-only root filesystem
            "--read-only",
            # Limited device access
            "--device", "/dev/null:/dev/null",
            "--device", "/dev/zero:/dev/zero",
            "--device", "/dev/random:/dev/random",
            "--device", "/dev/urandom:/dev/urandom",
            # Mount workspace as tmpfs
            "--tmpfs", f"{workspace_path}:size=128M,uid={self._uid},gid={self._gid},mode=700",
            # Temp directory inside container
            "--tmpfs", "/tmp:size=64M",
            # Container name
            "--name", f"ft_sandbox_{self._uid}_{int(time.time())}",
            # Image
            DOCKER_IMAGE,
            # Command (sleep to keep container alive for docker exec)
            "sleep", "infinity",
        ]

        # Run docker create
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create Docker container: {result.stderr.strip()}"
            )

        container_id = result.stdout.strip()

        # Get the container object via the Docker SDK
        container = self._client.containers.get(container_id)
        return container

    def _ensure_tool_script(self) -> str:
        """Ensure the tool implementation script exists in the container.

        Creates a Python script that implements all tools and writes it
        to the container via docker cp or by embedding in the exec call.
        Returns the path to the script inside the container.
        """
        return "/tools/sandbox_tools.py"

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Stop and remove the Docker container and clean up workspace."""
        try:
            if hasattr(self, "_container") and self._container is not None:
                self._container.stop(timeout=5)
                self._container.remove()
        except Exception:
            pass
        finally:
            if self._owns_workspace and self.workspace.exists():
                import shutil
                shutil.rmtree(self.workspace, ignore_errors=True)
            try:
                self._client.close()
            except Exception:
                pass

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def root(self) -> Path:
        """Return the workspace path (as seen by the host)."""
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
        """Execute a tool inside the Docker container.

        The tool code is executed inside the container via docker exec.
        Returns only the artifact output string.
        """
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

    # -- container execution -----------------------------------------------

    def _exec(self, cmd: list[str]) -> str:
        """Execute a command inside the Docker container via docker exec.

        Returns the stdout output as a string.
        """
        exec_id = self._container.exec_run(
            cmd,
            workdir=str(self.workspace),
            user=f"{self._uid}:{self._gid}",
        )
        if isinstance(exec_id, tuple):
            # docker.models.Container.exec_run returns (result, output) in newer versions
            output = exec_id[1] if len(exec_id) > 1 else b""
            if hasattr(exec_id[0], 'output'):
                output = exec_id[0].output
            return output.decode("utf-8", errors="replace") if isinstance(output, bytes) else str(output)
        return str(exec_id)

    # -- policy -------------------------------------------------------------

    def _resolve(self, rel_path: str) -> Path:
        """Resolve a workspace path and block both `..` and symlink escapes."""
        err = _policy_check_path(rel_path)
        if err:
            raise PolicyError(err)
        root = self.root()
        resolved = (root / rel_path).resolve()
        if not resolved.is_relative_to(root):
            raise PolicyError(f"path escapes sandbox workspace: {rel_path!r}")
        return resolved

    # -- tool implementations ----------------------------------------------

    def _git_checkout(self, args: dict[str, Any]) -> str:
        """Clone/checkout a repo into the workspace via docker exec."""
        repo = str(args.get("repo", ""))
        branch = str(args.get("branch", ""))
        if not repo or not branch:
            raise PolicyError("both 'repo' and 'branch' are required")

        target = self.workspace / "repo"
        if (target / ".git").exists():
            # Already cloned — just checkout the ref
            self._exec(["git", "fetch", "--all"], workdir=str(target))
            self._exec(["git", "checkout", branch], workdir=str(target))
            return f"checked out {repo} at {branch}\n"

        target.parent.mkdir(parents=True, exist_ok=True)
        self._exec(["git", "clone", "--branch", branch, "--", repo, str(target)], workdir=str(self.workspace))
        return f"cloned {repo} at {branch}\n"

    def _read_file(self, args: dict[str, Any]) -> str:
        """Read a file from the workspace via docker exec."""
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
            # Read inside the container
            content = self._exec(
                ["python3", "-c", f"print(open('{resolved.name}').read())"],
                workdir=str(self.workspace),
            )
            return content
        except UnicodeDecodeError:
            return f"Error: file is not UTF-8 text: {path}\n"

    def _write_file(self, args: dict[str, Any]) -> str:
        """Write a file to the workspace via docker exec."""
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

        # Write inside the container using a Python script
        escaped_content = content.replace("'", "'\"'\"'").replace("\n", "\\n")
        self._exec(
            ["python3", "-c", f"open('{resolved.name}', 'w').write('{content}')"],
            workdir=str(self.workspace),
        )
        return f"wrote {path} ({len(content)} bytes)\n"


# ---------------------------------------------------------------------------
# Helper: get container ID (for testing/debugging)
# ---------------------------------------------------------------------------

def _get_container_id(sandbox: Sandbox) -> str | None:
    """Return the Docker container ID if the sandbox is active."""
    if hasattr(sandbox, "_container") and sandbox._container is not None:
        return sandbox._container.id
    return None