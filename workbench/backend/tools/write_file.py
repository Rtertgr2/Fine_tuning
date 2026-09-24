"""Real file writer (T2.2) — writes files to the sandbox workspace.

Enforces the same policy as the sandbox: path traversal blocked, .env/secret
files blocked. Always writes the full file content (never a diff).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.tools.sandbox import _policy_check_path


class PolicyError(Exception):
    pass


def write_file(workspace: Path, args: dict[str, Any]) -> str:
    """Write a file to the workspace.

    Returns a status message. Raises PolicyError on policy violation.
    """
    path = str(args.get("path", ""))
    content = args.get("content")

    if not path:
        raise PolicyError("'path' is required")
    if not isinstance(content, str):
        raise PolicyError("'content' must be a string (full file content, not a diff)")

    err = _policy_check_path(path)
    if err:
        raise PolicyError(err)

    root = workspace.resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise PolicyError(f"path escapes sandbox workspace: {path!r}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return f"wrote {path} ({len(content)} bytes)\n"
