"""The 3 agent tools (real implementations land with the Phase 2 sandbox).

Schemas here are the single source of truth for:
- validator rule T2 (name must be registered)
- validator rule T3 (arguments complete + correctly typed)
- the `tools` field injected into tool_use chat templates
"""

from __future__ import annotations

from typing import Any

TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "git_checkout": {
        "type": "function",
        "function": {
            "name": "git_checkout",
            "description": "Check out a repository at a branch or ref into the sandbox workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository URL or path to clone/check out.",
                    },
                    "branch": {
                        "type": "string",
                        "description": "Branch, tag or commit ref to check out.",
                    },
                },
                "required": ["repo", "branch"],
            },
        },
    },
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path of the file relative to the workspace root.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write the full contents of a file, replacing it if it exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path of the file relative to the workspace root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete file content, never a diff or partial fragment.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
}

_JSON_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def tool_names() -> list[str]:
    return sorted(TOOL_REGISTRY)


def tool_schemas() -> list[dict[str, Any]]:
    """List form for the chat template's `tools` argument."""
    return [TOOL_REGISTRY[n] for n in tool_names()]


def check_arguments(name: str, arguments: dict[str, Any]) -> list[str]:
    """Return human-readable problems with a call's arguments (empty = ok)."""
    problems: list[str] = []
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return [f"unknown tool {name!r}"]
    params = entry["function"]["parameters"]
    props: dict[str, dict] = params.get("properties", {})
    required: list[str] = params.get("required", [])

    for req in required:
        if req not in arguments:
            problems.append(f"missing required argument {req!r}")
    for key, value in arguments.items():
        spec = props.get(key)
        if spec is None:
            problems.append(f"unknown argument {key!r}")
            continue
        expected = _JSON_TYPE_MAP.get(spec.get("type", ""))
        if expected and not isinstance(value, expected):
            problems.append(
                f"argument {key!r} should be {spec['type']}, got {type(value).__name__}"
            )
    return problems
