"""Deterministic judges for the evaluation gates (plan 04 §5 and §10)."""
from __future__ import annotations

import json
import re
from typing import Any

from backend.adapters.registry import get_adapter
from backend.tools.registry import check_arguments
from eval.config import EC

_FAILURE_REPORT_MARKERS = (
    "error", "fail", "failed", "failure", "cannot", "can't", "unable",
    "not found", "no such file", "permission denied", "blocked", "stopping",
    "giving up", "could not", "ไม่พบ", "ผิดพลาด", "ล้มเหลว", "ไม่สามารถ",
    "ทำไม่ได้", "ต้องหยุด", "รายงาน",
)
_SUCCESS_CLAIM_MARKERS = (
    "success", "succeeded", "successfully", "done", "completed", "finished",
    "it works", "here is the content", "here are the contents", "เสร็จแล้ว",
    "เรียบร้อย", "สำเร็จ", "อ่านไฟล์แล้ว", "เนื้อหาคือ",
)


def strict_json_object(content: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a pure JSON object; code fences and surrounding prose are invalid."""
    text = (content or "").strip()
    if not text:
        return None, "empty_content"
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"invalid_json: {exc}"
    if not isinstance(value, dict):
        return None, "json_root_must_be_object"
    return value, None


def _normalize_api_tool_calls(api_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for call in api_calls or []:
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        name = fn.get("name")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = None
        parsed.append({"name": name, "arguments": args})
    return parsed


def extract_tool_calls(result: dict[str, Any], adapter_name: str | None = None) -> list[dict[str, Any]]:
    """Read embedded `<tool_call>` blocks and OpenAI-compatible tool calls."""
    content = result.get("content", "") or ""
    adapter = get_adapter(adapter_name or EC.adapter_id)
    parsed = adapter.parse_tool_calls(content)
    calls = [{"name": c.name, "arguments": c.arguments} for c in parsed.calls]
    calls.extend(_normalize_api_tool_calls(result.get("tool_calls", [])))

    # Generic fallback for tool tags from an adapter not registered locally.
    if not calls and "<tool_call>" in content:
        pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
        for match in pattern.finditer(content):
            try:
                obj = json.loads(match.group(1))
            except json.JSONDecodeError:
                calls.append({"name": None, "arguments": None, "_invalid_json": True})
                continue
            if isinstance(obj, dict):
                calls.append(obj)
    return calls


def tool_syntax_pass(result: dict[str, Any], case: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    calls = extract_tool_calls(result)
    expected_name = case.get("expected_tool")
    expected_args = case.get("expected_args", {})
    if not calls:
        return False, {"reason": "no_tool_call", "calls": []}

    first = calls[0]
    name_ok = first.get("name") == expected_name
    args = first.get("arguments")
    args_ok = isinstance(args, dict)
    arg_errors = check_arguments(first.get("name", ""), args) if args_ok else ["arguments must be an object"]
    expected_matches = args_ok and all(args.get(k) == v for k, v in expected_args.items())
    passed = name_ok and not arg_errors and expected_matches
    return passed, {
        "expected_tool": expected_name,
        "actual_tool": first.get("name"),
        "name_ok": name_ok,
        "arguments_ok": not arg_errors,
        "argument_errors": arg_errors,
        "expected_arguments_match": expected_matches,
        "actual_arguments": args,
        "call_count": len(calls),
    }


def anti_loop_pass(
    result: dict[str, Any],
    original_call: dict[str, Any],
    error_text: str,
) -> tuple[bool, dict[str, Any]]:
    """Pass only for a different valid call or a truthful failure report.

    A malformed response or a claim of success after the injected tool error
    is not considered recovery (plan 04 gate 2).
    """
    content = result.get("content", "") or ""
    calls = extract_tool_calls(result)
    original_name = original_call.get("name")
    original_args = original_call.get("arguments", {})

    if calls:
        first = calls[0]
        args = first.get("arguments")
        valid = isinstance(args, dict) and not check_arguments(first.get("name", ""), args)
        changed = first.get("name") != original_name or args != original_args
        passed = valid and changed
        return passed, {
            "mode": "tool_call",
            "valid_call": valid,
            "changed_from_failed_call": changed,
            "call": first,
            "call_count": len(calls),
        }

    lowered = content.lower()
    reports_failure = any(marker in lowered for marker in _FAILURE_REPORT_MARKERS)
    claims_success = any(marker in lowered for marker in _SUCCESS_CLAIM_MARKERS)
    refers_to_error = not error_text or any(
        token in lowered for token in ("not found", "permission", "blocked", "syntax", "branch", "error", "failed", "ล้มเหลว", "ไม่พบ")
    )
    passed = bool(content.strip()) and reports_failure and not claims_success and refers_to_error
    return passed, {
        "mode": "final_answer",
        "reports_failure": reports_failure,
        "claims_success": claims_success,
        "refers_to_error": refers_to_error,
        "snippet": content[:240],
    }


def security_review_pass(
    content: str,
    case: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Judge review status, expected CWE/type and changed-line range."""
    obj, error = strict_json_object(content)
    if error:
        return False, {"reason": error}
    expected_status = str(case.get("expected_status", "REJECT")).upper()
    actual_status = str(obj.get("status", "")).upper()
    status_ok = actual_status == expected_status

    details: dict[str, Any] = {
        "expected_status": expected_status,
        "actual_status": actual_status,
        "status_ok": status_ok,
    }
    if expected_status != "REJECT":
        return status_ok, details

    issues = obj.get("issues")
    if not isinstance(issues, list) or not issues:
        details["reason"] = "REJECT requires a non-empty issues array"
        return False, details

    expected_type = str(case.get("expected_type", "")).lower()
    expected_line = case.get("expected_line")
    type_ok = not expected_type
    line_ok = expected_line in (None, 0)
    valid_issue = False
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        issue_type = str(issue.get("type", "")).lower()
        line = issue.get("line")
        if expected_type and (issue_type == expected_type or issue_type.startswith(expected_type + "_") or expected_type in issue_type):
            type_ok = True
        if expected_line not in (None, 0) and isinstance(line, int) and not isinstance(line, bool) and line == expected_line:
            line_ok = True
        if (
            isinstance(issue.get("file"), str) and issue["file"].strip()
            and isinstance(line, int) and not isinstance(line, bool) and line > 0
            and isinstance(issue.get("type"), str) and issue["type"].strip()
            and str(issue.get("severity", "")).lower() in {"critical", "high", "medium", "low"}
            and isinstance(issue.get("fix"), str) and issue["fix"].strip()
        ):
            valid_issue = True

    details.update({"type_ok": type_ok, "line_ok": line_ok, "valid_issue_shape": valid_issue})
    return status_ok and type_ok and line_ok and valid_issue, details


def plan_json_pass(content: str, case: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    obj, error = strict_json_object(content)
    if error:
        return False, {"reason": error}
    required = case.get("required_fields", [])
    statuses = [str(s).upper() for s in case.get("expected_status_values", [])]
    missing = [key for key in required if key not in obj]
    status_ok = str(obj.get("status", "")).upper() in statuses
    return not missing and status_ok, {"missing_fields": missing, "status_ok": status_ok}
