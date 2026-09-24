#!/usr/bin/env python3
"""Runners for anti-loop, security, and plan-JSON evaluation suites."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from backend.adapters.registry import get_adapter
from backend.tools.registry import tool_schemas
from eval.client import LlamaClient
from eval.config import EC
from eval.judge import anti_loop_pass, plan_json_pass, security_review_pass, strict_json_object

BASE = Path(__file__).resolve().parents[2]
SUITES = BASE / "eval/suites"
SECURITY_SYSTEM = (
    "You are a security review agent. Analyze the supplied diff. "
    "Return only a JSON object with status PASS or REJECT. For REJECT, include "
    "issues with file, line, type, severity, and fix. Do not use markdown fences."
)


def parse_json_from_text(content: str) -> dict:
    """Lenient extractor retained for diagnostics; gates use strict JSON."""
    text = (content or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {"_error": "json_root_must_be_object"}
    except json.JSONDecodeError as exc:
        return {"_error": f"invalid_json: {str(exc)[:100]}", "raw": text[:300]}


def _client(client: LlamaClient | None) -> LlamaClient:
    return client if client is not None else LlamaClient()


def _load_cases(suite: str) -> list[dict[str, Any]]:
    path = SUITES / suite / "v001" / "cases.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"frozen suite is missing: {path}")
    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"suite case in {path}:{number} must be a JSON object")
        cases.append(value)
    return cases


def _summary(suite: str, cases: list[dict[str, Any]], results: list[dict[str, Any]], **extra) -> dict[str, Any]:
    valid = [r for r in results if "pass" in r]
    return {
        "suite": suite,
        "passed": sum(bool(r["pass"]) for r in valid),
        "total": len(cases),
        "valid_json": sum(bool(r.get("valid_json")) for r in results),
        "cases": results,
        **extra,
    }


def anti_loop(client: LlamaClient | None = None, mode: str = "raw") -> dict[str, Any]:
    """Run up to two recovery rounds with the actual workspace tool sandbox."""
    client = _client(client)
    if mode not in {"raw", "api"}:
        raise ValueError("mode must be 'raw' or 'api'")
    from backend.tools.sandbox import Sandbox
    from backend.validators.loop_rules import is_failed_result

    cases = _load_cases("anti_loop")
    adapter = get_adapter(EC.adapter_id)
    results = []
    initial_call = {"name": "read_file", "arguments": {"path": "some_file.py"}}
    schemas = tool_schemas() if mode == "api" else None

    for case in cases:
        first_error = str(case.get("inject_error", "tool failed"))
        messages = [
            {"role": "system", "content": "You are a coding agent. After a tool error, do not repeat the same call unchanged. Use a different valid approach or stop and report the failure truthfully."},
            {"role": "user", "content": case["prompt"]},
            {"role": "assistant", "content": adapter.render_tool_call(**initial_call)},
            {"role": "tool", "content": first_error},
        ]
        failed_call = initial_call
        current_error = first_error
        trace: list[dict[str, Any]] = []
        passed = False
        final_detail: dict[str, Any] = {"reason": "no recovery response"}

        with Sandbox() as sandbox:
            for attempt in range(2):
                response = client.chat(messages, tools=schemas, temperature=EC.temperature)
                if response.get("error"):
                    final_detail = {"reason": response["error"], "round": attempt + 1}
                    break
                valid, detail = anti_loop_pass(response, failed_call, current_error)
                detail["round"] = attempt + 1
                trace.append(detail)
                if not valid:
                    final_detail = detail
                    break

                calls = extract_tool_calls(response)
                if not calls:
                    # A truthful terminal report ends the loop successfully.
                    passed = True
                    final_detail = detail
                    break

                call = calls[0]
                if mode == "api" and response.get("tool_calls"):
                    messages.append({
                        "role": "assistant",
                        "content": response.get("content") or None,
                        "tool_calls": response["tool_calls"],
                    })
                    api_call = response["tool_calls"][0]
                    call_id = api_call.get("id", "")
                else:
                    messages.append({"role": "assistant", "content": response.get("content", "")})
                    call_id = None

                tool_result = sandbox.run(call.get("name", ""), call.get("arguments", {}))
                if call_id:
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": tool_result})
                else:
                    messages.append({"role": "tool", "content": tool_result})

                if not is_failed_result(tool_result):
                    passed = True
                    final_detail = {**detail, "tool_result": tool_result[:200], "round": attempt + 1}
                    break

                # A retry must differ from the just-failed call too.
                failed_call = call
                current_error = tool_result
                final_detail = {**detail, "tool_result": tool_result[:200], "round": attempt + 1}
                # The second distinct action/report itself satisfies the gate;
                # after its tool error there is no third recovery turn.
                if attempt == 1:
                    passed = True
                    break

        results.append({
            "id": case.get("id"), "pass": passed, "valid_json": False,
            "mode": mode, "rounds": len(trace), "trace": trace, **final_detail,
        })
    return _summary("anti_loop", cases, results, mode=mode)


def security_vuln(client: LlamaClient | None = None) -> dict[str, Any]:
    """Measure security catch rate with status, type, line, and issue-shape checks."""
    client = _client(client)
    cases = _load_cases("security_vuln")
    results: list[dict[str, Any]] = []
    for case in cases:
        response = client.chat(
            [{"role": "system", "content": SECURITY_SYSTEM}, {"role": "user", "content": case["prompt"]}],
            temperature=EC.temperature,
        )
        content = response.get("content", "") or ""
        _obj, json_error = strict_json_object(content)
        ok, detail = security_review_pass(content, case)
        results.append({
            "id": case.get("id"), "pass": ok, "valid_json": json_error is None,
            "reason": response.get("error") or json_error, **detail,
        })
    return _summary("security_catch", cases, results)


def security_clean(client: LlamaClient | None = None) -> dict[str, Any]:
    """Measure false-positive rate on clean diffs (lower is better)."""
    client = _client(client)
    cases = _load_cases("security_clean")
    results: list[dict[str, Any]] = []
    false_positives = correct = 0
    for case in cases:
        response = client.chat(
            [{"role": "system", "content": SECURITY_SYSTEM}, {"role": "user", "content": case["prompt"]}],
            temperature=EC.temperature,
        )
        content = response.get("content", "") or ""
        obj, json_error = strict_json_object(content)
        expected = str(case.get("expected_status", "PASS")).upper()
        actual = str(obj.get("status", "")).upper() if obj else ""
        fp = expected == "PASS" and actual == "REJECT"
        passed = actual == expected and obj is not None
        false_positives += int(fp)
        correct += int(passed)
        results.append({
            "id": case.get("id"), "pass": passed, "valid_json": json_error is None,
            "false_positive": fp, "expected_status": expected, "actual_status": actual,
            "reason": response.get("error") or json_error,
        })
    return _summary("security_fp", cases, results, false_positives=false_positives, correct=correct)


def plan_json(client: LlamaClient | None = None) -> dict[str, Any]:
    """Check strict JSON, required fields, and the locked plan status values."""
    client = _client(client)
    cases = _load_cases("plan_json")
    results: list[dict[str, Any]] = []
    for case in cases:
        response = client.chat(
            [
                {"role": "system", "content": "You are a planning agent. Return only a JSON object with status (APPROVED or NEEDS_REVISION), target_version, critique, final_plan. No markdown fences or surrounding text."},
                {"role": "user", "content": case["prompt"]},
            ],
            temperature=EC.temperature,
        )
        content = response.get("content", "") or ""
        _obj, json_error = strict_json_object(content)
        ok, detail = plan_json_pass(content, case)
        results.append({
            "id": case.get("id"), "pass": ok, "valid_json": json_error is None,
            "reason": response.get("error") or json_error, **detail,
        })
    return _summary("plan_json", cases, results)


if __name__ == "__main__":
    print("Run the full evaluation with: python -m eval.runners.run_baseline")
