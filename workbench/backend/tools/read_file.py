"""Real file reader (T2.2) — reads files from the sandbox workspace.

Enforces the same policy as the sandbox: path traversal blocked, .env/secret
files blocked, oversized files blocked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.tools.sandbox import MAX_FILE_BYTES, _policy_check_path


class PolicyError(Exception):
    pass


def read_file(workspace: Path, args: dict[str, Any]) -> str:
    """Read a file from the workspace.

    Returns the file contents, or an error message if the file doesn't exist
    or violates policy.
    """
    path = str(args.get("path", ""))
    if not path:
        raise PolicyError("'path' is required")

    err = _policy_check_path(path)
    if err:
        raise PolicyError(err)

    resolved = (workspace / path).resolve()
    if not resolved.exists():
        return f"Error: no such file or directory: {path}\n"
    if not resolved.is_file():
        return f"Error: not a regular file: {path}\n"

    size = resolved.stat().st_size
    if size > MAX_FILE_BYTES:
        raise PolicyError(f"file exceeds size limit ({size} bytes)")

    try:
        return resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: file is not UTF-8 text: {path}\n"
