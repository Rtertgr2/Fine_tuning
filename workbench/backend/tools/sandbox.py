"""Re-export Docker-based sandbox from backend.app.services.sandbox_runtime.

This module preserves the import path `from backend.tools.sandbox import ...`
for all existing callers (write_file.py, read_file.py, tool_generator.py,
loop_generator.py, test_pipeline.py) while delegating to the Docker-based
implementation in sandbox_runtime.py.
"""
from backend.app.services.sandbox_runtime import (
    Sandbox,
    Trajectory,
    ToolCallRecord,
    PolicyError,
    MAX_FILE_BYTES,
    _policy_check_path,
    BLOCKED_FILENAMES,
    BLOCKED_EXTENSIONS,
)

__all__ = [
    "Sandbox",
    "Trajectory",
    "ToolCallRecord",
    "PolicyError",
    "MAX_FILE_BYTES",
    "_policy_check_path",
    "BLOCKED_FILENAMES",
    "BLOCKED_EXTENSIONS",
]
