"""Review queue database schema and operations (T2.9)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

REVIEW_QUEUE_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_queue (
    id              TEXT PRIMARY KEY,
    category        TEXT NOT NULL CHECK (category IN ('tool','loop','plan','sec')),
    example_data_json TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','approved','rejected')),
    reason          TEXT,
    reviewer        TEXT,
    group_id        TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_review_status ON review_queue(status);
CREATE INDEX IF NOT EXISTS idx_review_category ON review_queue(category);
CREATE INDEX IF NOT EXISTS idx_review_group ON review_queue(group_id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def list_pending(
    conn: sqlite3.Connection,
    category: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[int, list[dict[str, Any]]]:
    """List pending review items, newest first."""
    where = ["status = 'pending'"]
    params: list[Any] = []
    if category:
        where.append("category = ?")
        params.append(category)

    clause = f"WHERE {' AND '.join(where)}"
    total = conn.execute(
        f"SELECT COUNT(*) AS n FROM review_queue {clause}", params
    ).fetchone()["n"]

    rows = conn.execute(
        f"SELECT * FROM review_queue {clause} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        [*params, limit, offset],
    ).fetchall()

    return total, [_row_to_dict(r) for r in rows]


def get_item(conn: sqlite3.Connection, item_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM review_queue WHERE id = ?", (item_id,)).fetchone()
    return _row_to_dict(row) if row else None


def approve(
    conn: sqlite3.Connection,
    item_id: str,
    reviewer: str | None = None,
) -> dict[str, Any] | None:
    """Approve a review item: move to examples table, mark approved."""
    row = conn.execute("SELECT * FROM review_queue WHERE id = ?", (item_id,)).fetchone()
    if row is None:
        return None

    example_data = json.loads(row["example_data_json"])
    from backend.app import ids
    from backend.adapters.base import content_hash

    ex_id = ids.example_id()
    ts = now_iso()
    messages = example_data.get("messages", [])
    tools = example_data.get("tools")
    h = content_hash(messages, tools)

    conn.execute(
        """INSERT INTO examples
           (id, category, messages_json, tools_json, source, group_id, status,
            content_hash, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            ex_id,
            row["category"],
            json.dumps(messages, ensure_ascii=False),
            json.dumps(tools, ensure_ascii=False) if tools else None,
            "generated",
            row["group_id"],
            "approved",
            h,
            ts,
            ts,
        ),
    )

    conn.execute(
        """UPDATE review_queue
           SET status='approved', reviewer=?, updated_at=?
           WHERE id=?""",
        (reviewer, ts, item_id),
    )
    conn.commit()
    return get_item(conn, item_id)


def reject(
    conn: sqlite3.Connection,
    item_id: str,
    reason: str,
    reviewer: str | None = None,
) -> dict[str, Any] | None:
    """Reject a review item with a reason."""
    row = conn.execute("SELECT * FROM review_queue WHERE id = ?", (item_id,)).fetchone()
    if row is None:
        return None

    ts = now_iso()
    conn.execute(
        """UPDATE review_queue
           SET status='rejected', reason=?, reviewer=?, updated_at=?
           WHERE id=?""",
        (reason, reviewer, ts, item_id),
    )
    conn.commit()
    return get_item(conn, item_id)


def edit_and_approve(
    conn: sqlite3.Connection,
    item_id: str,
    edited_data: dict[str, Any],
    reviewer: str | None = None,
) -> dict[str, Any] | None:
    """Edit the example data and approve it."""
    row = conn.execute("SELECT * FROM review_queue WHERE id = ?", (item_id,)).fetchone()
    if row is None:
        return None

    # Update the review queue item with edited data
    ts = now_iso()
    conn.execute(
        """UPDATE review_queue
           SET example_data_json=?, updated_at=?
           WHERE id=?""",
        (json.dumps(edited_data, ensure_ascii=False), ts, item_id),
    )

    # Then approve with the edited data
    from backend.app import ids
    from backend.adapters.base import content_hash

    ex_id = ids.example_id()
    messages = edited_data.get("messages", [])
    tools = edited_data.get("tools")
    h = content_hash(messages, tools)

    conn.execute(
        """INSERT INTO examples
           (id, category, messages_json, tools_json, source, group_id, status,
            content_hash, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            ex_id,
            row["category"],
            json.dumps(messages, ensure_ascii=False),
            json.dumps(tools, ensure_ascii=False) if tools else None,
            "generated",
            row["group_id"],
            "approved",
            h,
            ts,
            ts,
        ),
    )

    conn.execute(
        """UPDATE review_queue
           SET status='approved', reviewer=?, updated_at=?
           WHERE id=?""",
        (reviewer, ts, item_id),
    )
    conn.commit()
    return get_item(conn, item_id)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "category": row["category"],
        "example_data": json.loads(row["example_data_json"]),
        "status": row["status"],
        "reason": row["reason"],
        "reviewer": row["reviewer"],
        "group_id": row["group_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
