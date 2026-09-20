"""Structured logging service — write agent events to SQLite.

Log schema per Plan/06 §4:
  session_id, ts, model_version, state, event, data, project
Redacts secrets at write time before persisting.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from backend.app.services.redaction import redact, redact_dict


VALID_STATES = ("plan", "code", "review", "idle")
VALID_EVENTS = (
    "message", "tool_call", "tool_result", "circuit_breaker",
    "verdict", "override", "outcome", "cancel", "undo",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def insert_event(
    conn: sqlite3.Connection,
    session_id: str,
    model_version: str,
    state: str,
    event: str,
    data: dict[str, Any],
    project: str | None = None,
) -> str:
    """Insert a single redacted event. Returns the row id."""
    if state not in VALID_STATES:
        raise ValueError(f"invalid state: {state}")
    if event not in VALID_EVENTS:
        raise ValueError(f"invalid event: {event}")

    cleaned_data, redaction_info = redact_dict(data)

    # Also redact project name if provided
    clean_project = redact(project).text if project else None

    conn.execute(
        """INSERT INTO activity_log
           (session_id, ts, model_version, state, event, data, project,
            redaction_count, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            session_id,
            _now(),
            model_version,
            state,
            event,
            json.dumps(cleaned_data, ensure_ascii=False),
            clean_project,
            redaction_info.redaction_count if redaction_info else 0,
            _now(),
        ),
    )
    conn.commit()
    return session_id


def insert_batch(
    conn: sqlite3.Connection,
    events: list[dict[str, Any]],
) -> int:
    """Insert a batch of events. Each dict must have session_id,
    model_version, state, event, data. Returns count inserted."""
    now = _now()
    rows = []
    for ev in events:
        state = ev["state"]
        event = ev["event"]
        if state not in VALID_STATES:
            raise ValueError(f"invalid state: {state}")
        if event not in VALID_EVENTS:
            raise ValueError(f"invalid event: {event}")
        cleaned_data, redaction_info = redact_dict(ev["data"])
        clean_project = redact(ev.get("project", "")).text if ev.get("project") else None
        rows.append((
            ev["session_id"],
            now,
            ev["model_version"],
            state,
            event,
            json.dumps(cleaned_data, ensure_ascii=False),
            clean_project,
            redaction_info.redaction_count if redaction_info else 0,
            now,
        ))
    conn.executemany(
        """INSERT INTO activity_log
           (session_id, ts, model_version, state, event, data, project,
            redaction_count, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    return len(rows)


def get_events_for_session(
    conn: sqlite3.Connection,
    session_id: str,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Get all events for a session, ordered by timestamp."""
    rows = conn.execute(
        """SELECT * FROM activity_log
           WHERE session_id = ?
           ORDER BY ts ASC
           LIMIT ?""",
        (session_id, limit),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def query_events(
    conn: sqlite3.Connection,
    session_id: str | None = None,
    model_version: str | None = None,
    state: str | None = None,
    event: str | None = None,
    project: str | None = None,
    since: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[int, list[dict[str, Any]]]:
    """Query activity log with filters."""
    where: list[str] = []
    params: list[Any] = []
    if session_id:
        where.append("session_id = ?")
        params.append(session_id)
    if model_version:
        where.append("model_version = ?")
        params.append(model_version)
    if state:
        where.append("state = ?")
        params.append(state)
    if event:
        where.append("event = ?")
        params.append(event)
    if project:
        where.append("project = ?")
        params.append(project)
    if since:
        where.append("ts >= ?")
        params.append(since)

    clause = f"WHERE {' AND '.join(where)}" if where else ""
    total = conn.execute(
        f"SELECT COUNT(*) AS n FROM activity_log {clause}", params
    ).fetchone()["n"]

    rows = conn.execute(
        f"SELECT * FROM activity_log {clause} ORDER BY ts DESC LIMIT ? OFFSET ?",
        [*params, limit, offset],
    ).fetchall()
    return total, [_row_to_dict(r) for r in rows]


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "ts": row["ts"],
        "model_version": row["model_version"],
        "state": row["state"],
        "event": row["event"],
        "data": json.loads(row["data"]),
        "project": row["project"],
        "redaction_count": row["redaction_count"],
        "created_at": row["created_at"],
    }
