"""Case detector — scan activity_log for failure signals and create case_queue entries.

Signals per Plan/06 §5:
  - Circuit breaker triggered (loop)
  - Same tool call repeated 2+ times (loop)
  - JSON/tool_call malformed (syntax)
  - No change after tool error (loop)
  - Reviewer PASS but test failed (false negative)
  - Reviewer REJECT but confirmed safe (false positive)
  - User cancellation/undo

Also samples successful cases (~20%) for replay.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from backend.app.services import activity_log as log_svc


@dataclass
class DetectedCase:
    """A detected case from activity log analysis."""
    session_id: str
    model_version: str
    case_type: str  # loop, syntax, false_negative, false_positive, cancel, success
    priority: int  # 1-5, 5 = highest
    signal: str  # human-readable description
    events: list[dict[str, Any]]  # relevant events
    project: str | None = None


# Case type constants
CASE_LOOP = "loop"
CASE_SYNTAX = "syntax"
CASE_FALSE_NEGATIVE = "false_negative"
CASE_FALSE_POSITIVE = "false_positive"
CASE_CANCEL = "cancel"
CASE_SUCCESS = "success"

PRIORITY_MAP = {
    CASE_LOOP: 4,
    CASE_SYNTAX: 3,
    CASE_FALSE_NEGATIVE: 5,
    CASE_FALSE_POSITIVE: 3,
    CASE_CANCEL: 2,
    CASE_SUCCESS: 1,
}


def detect_cases_for_session(
    conn: sqlite3.Connection,
    session_id: str,
) -> list[DetectedCase]:
    """Analyze a session's events and return detected cases."""
    events = log_svc.get_events_for_session(conn, session_id)
    if not events:
        return []

    cases: list[DetectedCase] = []
    model_version = events[0]["model_version"]
    project = events[0].get("project")

    # Check for circuit breaker
    cb_events = [e for e in events if e["event"] == "circuit_breaker"]
    if cb_events:
        cases.append(DetectedCase(
            session_id=session_id,
            model_version=model_version,
            case_type=CASE_LOOP,
            priority=PRIORITY_MAP[CASE_LOOP],
            signal=f"Circuit breaker triggered ({len(cb_events)} times)",
            events=events,
            project=project,
        ))

    # Check for repeated tool calls (same tool_call 2+ times consecutively)
    tool_calls = [e for e in events if e["event"] == "tool_call"]
    if len(tool_calls) >= 2:
        for i in range(len(tool_calls) - 1):
            curr = tool_calls[i]["data"]
            next_ = tool_calls[i + 1]["data"]
            if _same_tool_call(curr, next_):
                cases.append(DetectedCase(
                    session_id=session_id,
                    model_version=model_version,
                    case_type=CASE_LOOP,
                    priority=PRIORITY_MAP[CASE_LOOP],
                    signal=f"Repeated tool call: {_tool_call_desc(curr)}",
                    events=events,
                    project=project,
                ))
                break

    # Check for malformed JSON in tool calls
    for e in events:
        if e["event"] == "tool_call":
            if not _valid_tool_json(e["data"]):
                cases.append(DetectedCase(
                    session_id=session_id,
                    model_version=model_version,
                    case_type=CASE_SYNTAX,
                    priority=PRIORITY_MAP[CASE_SYNTAX],
                    signal=f"Malformed tool_call JSON: {str(e['data'])[:100]}",
                    events=events,
                    project=project,
                ))
                break

    # Check whether the assistant repeats the failed tool call after an error.
    # Compare structured (name, arguments), not the tool_result body against an
    # arbitrary assistant message — that comparison never detected real loops.
    for i, event in enumerate(events):
        if event["event"] != "tool_result" or not _is_error_result(event["data"]):
            continue
        previous_call = next(
            (e["data"] for e in reversed(events[:i]) if e["event"] == "tool_call"),
            None,
        )
        next_assistant = next(
            (e for e in events[i + 1:] if e["event"] == "message" and e["data"].get("role") == "assistant"),
            None,
        )
        next_call = _assistant_tool_call(next_assistant["data"]) if next_assistant else None
        if previous_call and next_call and _same_tool_call(previous_call, next_call):
            cases.append(DetectedCase(
                session_id=session_id,
                model_version=model_version,
                case_type=CASE_LOOP,
                priority=PRIORITY_MAP[CASE_LOOP],
                signal=f"No strategy change after tool error: {_tool_call_desc(previous_call)}",
                events=events,
                project=project,
            ))
            break

    # Check for reviewer false negative (PASS but test failed)
    verdicts = [e for e in events if e["event"] == "verdict"]
    outcomes = [e for e in events if e["event"] == "outcome"]
    for v in verdicts:
        if v["data"].get("verdict") == "PASS":
            # Check if any subsequent outcome shows failure
            for o in outcomes:
                if o["data"].get("success") is False:
                    cases.append(DetectedCase(
                        session_id=session_id,
                        model_version=model_version,
                        case_type=CASE_FALSE_NEGATIVE,
                        priority=PRIORITY_MAP[CASE_FALSE_NEGATIVE],
                        signal="Reviewer PASS but outcome failed",
                        events=events,
                        project=project,
                    ))
                    break

    # Check for reviewer false positive (REJECT but override confirmed safe)
    for v in verdicts:
        if v["data"].get("verdict") == "REJECT":
            # Check for override event confirming safe
            overrides = [e for e in events if e["event"] == "override"]
            for ov in overrides:
                if ov["data"].get("confirmed_safe") is True:
                    cases.append(DetectedCase(
                        session_id=session_id,
                        model_version=model_version,
                        case_type=CASE_FALSE_POSITIVE,
                        priority=PRIORITY_MAP[CASE_FALSE_POSITIVE],
                        signal="Reviewer REJECT but confirmed safe",
                        events=events,
                        project=project,
                    ))
                    break

    # Check for user cancellation/undo
    cancel_events = [e for e in events if e["event"] in ("cancel", "undo")]
    if cancel_events:
        cases.append(DetectedCase(
            session_id=session_id,
            model_version=model_version,
            case_type=CASE_CANCEL,
            priority=PRIORITY_MAP[CASE_CANCEL],
            signal=f"User {cancel_events[0]['event']}",
            events=events,
            project=project,
        ))

    # If no failure cases detected, this is a successful session
    if not cases:
        cases.append(DetectedCase(
            session_id=session_id,
            model_version=model_version,
            case_type=CASE_SUCCESS,
            priority=PRIORITY_MAP[CASE_SUCCESS],
            signal="Session completed successfully",
            events=events,
            project=project,
        ))

    return cases


def _same_tool_call(a: dict, b: dict) -> bool:
    """Check if two tool_call data dicts represent the same call."""
    return (
        a.get("name") == b.get("name")
        and a.get("arguments") == b.get("arguments")
    )


def _tool_call_desc(data: dict) -> str:
    name = data.get("name", "?")
    args = data.get("arguments", {})
    arg_str = ", ".join(f"{k}={v}" for k, v in list(args.items())[:3])
    return f"{name}({arg_str})"


def _valid_tool_json(data: dict) -> bool:
    """Check a logged tool call has a non-empty name and JSON-object args."""
    if not isinstance(data, dict):
        return False
    function = data.get("function") if isinstance(data.get("function"), dict) else data
    name = function.get("name")
    args = function.get("arguments")
    if not isinstance(name, str) or not name.strip():
        return False
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return False
    return isinstance(args, dict)


def _assistant_tool_call(data: dict) -> dict | None:
    """Extract a tool call from a structured event or assistant message."""
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("name"), str):
        args = data.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        return {"name": data["name"], "arguments": args} if isinstance(args, dict) else None
    api_calls = data.get("tool_calls")
    if isinstance(api_calls, list) and api_calls:
        first = api_calls[0]
        fn = first.get("function", {}) if isinstance(first, dict) else {}
        if isinstance(fn, dict):
            return _assistant_tool_call(fn)
    content = data.get("content", "")
    if isinstance(content, str) and content:
        from backend.adapters.registry import get_adapter
        from backend.app import config
        parsed = get_adapter(config.ACTIVE_ADAPTER).parse_tool_calls(content)
        if parsed.calls:
            call = parsed.calls[0]
            return {"name": call.name, "arguments": call.arguments}
    return None


def _is_error_result(data: dict) -> bool:
    """Check if a tool_result indicates an error."""
    if data.get("error"):
        return True
    if data.get("status") == "error":
        return True
    content = str(data.get("content", ""))
    return "error" in content.lower() or "exception" in content.lower()


def scan_all_sessions(
    conn: sqlite3.Connection,
    sample_success_rate: float = 0.2,
    since: str | None = None,
) -> list[DetectedCase]:
    """Scan all sessions in the activity log and detect cases.
    
    Args:
        conn: Database connection
        sample_success_rate: Fraction of successful cases to include (0.0-1.0)
        since: Only scan sessions after this ISO timestamp
    """
    # Get all unique session IDs
    query = "SELECT DISTINCT session_id FROM activity_log"
    params: list[Any] = []
    if since:
        query += " WHERE ts >= ?"
        params.append(since)
    
    rows = conn.execute(query, params).fetchall()
    all_cases: list[DetectedCase] = []
    
    for row in rows:
        session_cases = detect_cases_for_session(conn, row["session_id"])
        # Filter: only include success cases at the sampled rate
        filtered = []
        for case in session_cases:
            if case.case_type == CASE_SUCCESS:
                # Use hash of session_id for deterministic sampling
                import hashlib
                h = int(hashlib.md5(case.session_id.encode()).hexdigest(), 16)
                if (h % 100) / 100.0 > sample_success_rate:
                    continue
            filtered.append(case)
        all_cases.extend(filtered)
    
    # Sort by priority (highest first)
    all_cases.sort(key=lambda c: c.priority, reverse=True)
    return all_cases
