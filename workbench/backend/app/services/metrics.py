"""Active Learning metrics service — dashboard metrics per Plan/06 §9.

Metrics:
  - Circuit breaker rate per 100 sessions by model version
  - Tool error rate and recovery rate within 2 rounds
  - False positive/negative rates of reviewer
  - Task success rate and average time
  - Queue size, case age, type distribution
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any


def compute_metrics(conn: sqlite3.Connection) -> dict[str, Any]:
    """Compute all active learning metrics."""
    return {
        "circuit_breaker_rate": _circuit_breaker_rate(conn),
        "tool_error_metrics": _tool_error_metrics(conn),
        "reviewer_metrics": _reviewer_metrics(conn),
        "task_success_metrics": _task_success_metrics(conn),
        "queue_metrics": _queue_metrics(conn),
        "case_type_distribution": _case_type_distribution(conn),
    }


def _circuit_breaker_rate(conn: sqlite3.Connection) -> dict[str, Any]:
    """Circuit breaker rate per 100 sessions by model version."""
    rows = conn.execute(
        """SELECT model_version,
                  COUNT(DISTINCT session_id) AS total_sessions,
                  COUNT(CASE WHEN event = 'circuit_breaker' THEN 1 END) AS cb_events
           FROM activity_log
           GROUP BY model_version
           ORDER BY model_version"""
    ).fetchall()

    result = {}
    for row in rows:
        total = row["total_sessions"]
        cb = row["cb_events"]
        rate_per_100 = (cb / total * 100) if total > 0 else 0
        result[row["model_version"]] = {
            "total_sessions": total,
            "circuit_breaker_events": cb,
            "rate_per_100_sessions": round(rate_per_100, 2),
        }
    return result


def _tool_error_metrics(conn: sqlite3.Connection) -> dict[str, Any]:
    """Tool error rate and recovery rate within 2 rounds."""
    # Count tool calls and tool errors
    rows = conn.execute(
        """SELECT model_version,
                  COUNT(CASE WHEN event = 'tool_call' THEN 1 END) AS total_calls,
                  COUNT(CASE WHEN event = 'tool_result' AND data LIKE '%error%' THEN 1 END) AS errors
           FROM activity_log
           GROUP BY model_version"""
    ).fetchall()

    result = {}
    for row in rows:
        total = row["total_calls"]
        errors = row["errors"]
        error_rate = (errors / total * 100) if total > 0 else 0

        # Recovery: after tool error, next 2 events contain a different tool call
        recovery_count = _count_recoveries(conn, row["model_version"])
        recovery_rate = (recovery_count / errors * 100) if errors > 0 else 100

        result[row["model_version"]] = {
            "total_tool_calls": total,
            "tool_errors": errors,
            "error_rate_pct": round(error_rate, 2),
            "recoveries_within_2_rounds": recovery_count,
            "recovery_rate_pct": round(recovery_rate, 2),
        }
    return result


def _count_recoveries(conn: sqlite3.Connection, model_version: str) -> int:
    """Count tool errors followed by a different tool call within 2 rounds."""
    events = conn.execute(
        """SELECT session_id, event, data, ts
           FROM activity_log
           WHERE model_version = ?
           ORDER BY session_id, ts""",
        (model_version,),
    ).fetchall()

    recoveries = 0
    prev_error_session = None
    prev_error_idx = -1

    for i, ev in enumerate(events):
        if ev["event"] == "tool_result" and "error" in str(ev["data"]).lower():
            prev_error_session = ev["session_id"]
            prev_error_idx = i
        elif (ev["event"] == "tool_call" and prev_error_session == ev["session_id"]
              and 0 <= i - prev_error_idx <= 3):
            recoveries += 1
            prev_error_session = None

    return recoveries


def _reviewer_metrics(conn: sqlite3.Connection) -> dict[str, Any]:
    """False positive/negative rates of reviewer."""
    # False negative: verdict PASS but outcome failed
    # False positive: verdict REJECT but override confirmed safe
    rows = conn.execute(
        """SELECT model_version,
                  COUNT(CASE WHEN event = 'verdict' AND json_extract(data, '$.verdict') = 'PASS' THEN 1 END) AS pass_count,
                  COUNT(CASE WHEN event = 'verdict' AND json_extract(data, '$.verdict') = 'REJECT' THEN 1 END) AS reject_count,
                  COUNT(CASE WHEN event = 'outcome' AND json_extract(data, '$.success') = 0 THEN 1 END) AS failed_outcomes,
                  COUNT(CASE WHEN event = 'override' AND json_extract(data, '$.confirmed_safe') = 1 THEN 1 END) AS confirmed_safe_overrides
           FROM activity_log
           GROUP BY model_version"""
    ).fetchall()

    result = {}
    for row in rows:
        pass_count = row["pass_count"]
        reject_count = row["reject_count"]
        failed = row["failed_outcomes"]
        confirmed_safe = row["confirmed_safe_overrides"]

        fn_rate = (failed / pass_count * 100) if pass_count > 0 else 0
        fp_rate = (confirmed_safe / reject_count * 100) if reject_count > 0 else 0

        result[row["model_version"]] = {
            "reviewer_pass_count": pass_count,
            "reviewer_reject_count": reject_count,
            "false_negatives": failed,
            "false_positives": confirmed_safe,
            "false_negative_rate_pct": round(fn_rate, 2),
            "false_positive_rate_pct": round(fp_rate, 2),
        }
    return result


def _task_success_metrics(conn: sqlite3.Connection) -> dict[str, Any]:
    """Task success rate and average time."""
    rows = conn.execute(
        """SELECT model_version,
                  COUNT(DISTINCT session_id) AS total_sessions,
                  COUNT(CASE WHEN event = 'outcome' AND json_extract(data, '$.success') = 1 THEN 1 END) AS successes,
                  AVG(CASE WHEN event = 'outcome' THEN json_extract(data, '$.duration_ms') END) AS avg_duration_ms
           FROM activity_log
           GROUP BY model_version"""
    ).fetchall()

    result = {}
    for row in rows:
        total = row["total_sessions"]
        successes = row["successes"]
        success_rate = (successes / total * 100) if total > 0 else 0

        result[row["model_version"]] = {
            "total_sessions": total,
            "successful_sessions": successes,
            "success_rate_pct": round(success_rate, 2),
            "avg_duration_ms": round(row["avg_duration_ms"], 2) if row["avg_duration_ms"] else None,
        }
    return result


def _queue_metrics(conn: sqlite3.Connection) -> dict[str, Any]:
    """Queue size, case age."""
    total = conn.execute("SELECT COUNT(*) AS n FROM case_queue").fetchone()["n"]
    pending = conn.execute(
        "SELECT COUNT(*) AS n FROM case_queue WHERE status = 'pending'"
    ).fetchone()["n"]
    approved = conn.execute(
        "SELECT COUNT(*) AS n FROM case_queue WHERE status = 'approved'"
    ).fetchone()["n"]
    rejected = conn.execute(
        "SELECT COUNT(*) AS n FROM case_queue WHERE status = 'rejected'"
    ).fetchone()["n"]

    # Average age of pending cases
    now = datetime.now(timezone.utc)
    pending_rows = conn.execute(
        "SELECT created_at FROM case_queue WHERE status = 'pending'"
    ).fetchall()

    avg_age_hours = 0
    if pending_rows:
        ages = []
        for r in pending_rows:
            created = datetime.fromisoformat(r["created_at"])
            ages.append((now - created).total_seconds() / 3600)
        avg_age_hours = sum(ages) / len(ages)

    return {
        "total_cases": total,
        "pending": pending,
        "approved": approved,
        "rejected": rejected,
        "avg_pending_age_hours": round(avg_age_hours, 2),
    }


def _case_type_distribution(conn: sqlite3.Connection) -> dict[str, int]:
    """Distribution of cases by type."""
    rows = conn.execute(
        "SELECT case_type, COUNT(*) AS n FROM case_queue GROUP BY case_type ORDER BY n DESC"
    ).fetchall()
    return {row["case_type"]: row["n"] for row in rows}
