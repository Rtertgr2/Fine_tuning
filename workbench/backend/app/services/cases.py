"""Case queue management service — CRUD for detected cases and editor workflows."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from backend.app.services.case_detector import DetectedCase


CASE_STATUS_PENDING = "pending"
CASE_STATUS_APPROVED = "approved"
CASE_STATUS_REJECTED = "rejected"
CASE_STATUS_EDITING = "editing"
CASE_STATUS_MERGED = "merged"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def enqueue_case(
    conn: sqlite3.Connection,
    case: DetectedCase,
) -> str:
    """Add a detected case to the queue. Returns the case_id."""
    case_id = _generate_case_id()
    conn.execute(
        """INSERT INTO case_queue
           (case_id, session_id, model_version, case_type, priority,
            signal, events_json, project, status, reviewer, second_reviewer,
            rejection_reason, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            case_id,
            case.session_id,
            case.model_version,
            case.case_type,
            case.priority,
            case.signal,
            json.dumps(case.events, ensure_ascii=False),
            case.project,
            CASE_STATUS_PENDING,
            None,
            None,
            None,
            _now(),
            _now(),
        ),
    )
    conn.commit()
    return case_id


def enqueue_batch(
    conn: sqlite3.Connection,
    cases: list[DetectedCase],
) -> list[str]:
    """Enqueue multiple cases. Returns list of case_ids."""
    return [enqueue_case(conn, c) for c in cases]


def list_cases(
    conn: sqlite3.Connection,
    case_type: str | None = None,
    priority: int | None = None,
    status: str | None = None,
    model_version: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[int, list[dict[str, Any]]]:
    """List cases with optional filters."""
    where: list[str] = []
    params: list[Any] = []
    if case_type:
        where.append("case_type = ?")
        params.append(case_type)
    if priority is not None:
        where.append("priority = ?")
        params.append(priority)
    if status:
        where.append("status = ?")
        params.append(status)
    if model_version:
        where.append("model_version = ?")
        params.append(model_version)

    clause = f"WHERE {' AND '.join(where)}" if where else ""
    total = conn.execute(
        f"SELECT COUNT(*) AS n FROM case_queue {clause}", params
    ).fetchone()["n"]

    rows = conn.execute(
        f"SELECT * FROM case_queue {clause} ORDER BY priority DESC, created_at ASC LIMIT ? OFFSET ?",
        [*params, limit, offset],
    ).fetchall()
    return total, [_row_to_dict(r) for r in rows]


def get_case(conn: sqlite3.Connection, case_id: str) -> dict[str, Any] | None:
    """Get a single case by ID with full timeline."""
    row = conn.execute("SELECT * FROM case_queue WHERE case_id = ?", (case_id,)).fetchone()
    if row is None:
        return None
    case = _row_to_dict(row)
    # Add timeline (parsed events)
    case["timeline"] = json.loads(row["events_json"])
    return case


def start_editing(
    conn: sqlite3.Connection,
    case_id: str,
    reviewer: str,
) -> dict[str, Any] | None:
    """Mark a case as being edited."""
    row = conn.execute("SELECT * FROM case_queue WHERE case_id = ?", (case_id,)).fetchone()
    if row is None:
        return None
    conn.execute(
        """UPDATE case_queue
           SET status = ?, reviewer = ?, updated_at = ?
           WHERE case_id = ?""",
        (CASE_STATUS_EDITING, reviewer, _now(), case_id),
    )
    conn.commit()
    return get_case(conn, case_id)


def approve_case(
    conn: sqlite3.Connection,
    case_id: str,
    final_example: dict[str, Any] | None = None,
    second_reviewer: str | None = None,
) -> dict[str, Any] | None:
    """Approve a case. Sec cases require second reviewer."""
    row = conn.execute("SELECT * FROM case_queue WHERE case_id = ?", (case_id,)).fetchone()
    if row is None:
        return None

    # sec cases need second reviewer
    if row["case_type"] == "sec" and not second_reviewer:
        raise ValueError("sec cases require second_reviewer approval")

    conn.execute(
        """UPDATE case_queue
           SET status = ?, edited_example_json = ?, second_reviewer = ?, updated_at = ?
           WHERE case_id = ?""",
        (
            CASE_STATUS_APPROVED,
            json.dumps(final_example, ensure_ascii=False) if final_example else None,
            second_reviewer,
            _now(),
            case_id,
        ),
    )
    conn.commit()
    return get_case(conn, case_id)


def reject_case(
    conn: sqlite3.Connection,
    case_id: str,
    reason: str,
) -> dict[str, Any] | None:
    """Reject a case with a reason."""
    row = conn.execute("SELECT * FROM case_queue WHERE case_id = ?", (case_id,)).fetchone()
    if row is None:
        return None
    conn.execute(
        """UPDATE case_queue
           SET status = ?, rejection_reason = ?, updated_at = ?
           WHERE case_id = ?""",
        (CASE_STATUS_REJECTED, reason, _now(), case_id),
    )
    conn.commit()
    return get_case(conn, case_id)


def get_approved_cases(
    conn: sqlite3.Connection,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """Get all approved cases with their edited examples."""
    rows = conn.execute(
        """SELECT * FROM case_queue
           WHERE status = ?
           ORDER BY priority DESC
           LIMIT ?""",
        (CASE_STATUS_APPROVED, limit),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def _generate_case_id() -> str:
    """Generate a sortable case ID."""
    import time
    import os
    ts = int(time.time() * 1000)
    rand = int.from_bytes(os.urandom(4), "big")
    return f"c_{ts:012x}{rand:08x}"


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "session_id": row["session_id"],
        "model_version": row["model_version"],
        "case_type": row["case_type"],
        "priority": row["priority"],
        "signal": row["signal"],
        "project": row["project"],
        "status": row["status"],
        "reviewer": row["reviewer"],
        "second_reviewer": row["second_reviewer"],
        "rejection_reason": row["rejection_reason"],
        "edited_example_json": row["edited_example_json"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
