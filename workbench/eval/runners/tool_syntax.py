#!/usr/bin/env python3
"""Tool-call format and schema runner (T4.5 / plan 04 gate 1)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from backend.tools.registry import tool_schemas
from eval.client import LlamaClient
from eval.config import EC
from eval.judge import extract_tool_calls, tool_syntax_pass

BASE = Path(__file__).resolve().parents[2]
SUITES = BASE / "eval/suites"
SYSTEM = (
    "You are a coding agent. Use the supplied tools to perform the user's task. "
    "Return a tool call, not an explanation."
)


def parse_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse embedded calls without truncating nested JSON objects."""
    # Reuse the adapter's stateful tag parser (not a non-greedy brace regex).
    from backend.adapters.registry import get_adapter

    parsed = get_adapter(EC.adapter_id).parse_tool_calls(content or "")
    calls = [{"name": c.name, "arguments": c.arguments} for c in parsed.calls]
    if parsed.issues:
        calls.extend({"_error": issue.kind, "message": issue.message} for issue in parsed.issues)
    return calls


def has_tools_in_response(result: dict[str, Any]) -> bool:
    return bool(extract_tool_calls(result))


def evaluate(client: LlamaClient | None = None, mode: str = "api") -> dict[str, Any]:
    """Run the frozen suite using structured tool calls or raw embedded tags.

    The report records the mode so parity can be measured in both serving
    paths (plan 04 §6). Tool arguments are checked for required types and
    exact expected values, not merely for presence.
    """
    client = client or LlamaClient()
    if mode not in {"raw", "api"}:
        raise ValueError("mode must be 'raw' or 'api'")
    suite_path = SUITES / "tool_syntax/v001/cases.jsonl"
    cases = [json.loads(line) for line in suite_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results: list[dict[str, Any]] = []
    for case in cases:
        payload: dict[str, Any] = {"temperature": EC.temperature}
        if mode == "api":
            payload["tools"] = tool_schemas()
        response = client.chat(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": case["prompt"]},
            ],
            **payload,
        )
        passed, detail = tool_syntax_pass(response, case)
        results.append({
            "id": case.get("id"),
            "pass": passed,
            "reason": response.get("error"),
            **detail,
        })
    return {
        "suite": "tool_syntax",
        "mode": mode,
        "passed": sum(bool(r["pass"]) for r in results),
        "total": len(cases),
        "valid_json": sum(1 for r in results if r.get("arguments_ok")),
        "cases": results,
    }


if __name__ == "__main__":
    result = evaluate()
    rate = 100 * result["passed"] / max(result["total"], 1)
    gate = EC.gates[0]
    print(f"Tool syntax ({result['mode']}): {result['passed']}/{result['total']} = {rate:.1f}%")
    print(f"Gate: {'PASS' if rate >= gate.min_pct else 'FAIL'} (requires {gate.min_pct:.1f}%)")
