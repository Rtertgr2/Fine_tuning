"""Example CRUD + validation orchestration."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Sequence

from backend.adapters.base import content_hash
from backend.adapters.registry import get_adapter
from backend.app import ids, schemas
from backend.validators.base import Example as FwExample
from backend.validators.registry import Report, make_context, validate_example


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# row <-> model conversions
# ---------------------------------------------------------------------------


def row_to_framework(row: sqlite3.Row) -> FwExample:
    return FwExample(
        id=row["id"],
        category=row["category"],
        messages=json.loads(row["messages_json"]),
        tools=json.loads(row["tools_json"]) if row["tools_json"] else None,
        source=row["source"],
        group_id=row["group_id"],
        status=row["status"],
        content_hash=row["content_hash"],
    )


def row_to_out(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "category": row["category"],
        "messages": json.loads(row["messages_json"]),
        "tools": json.loads(row["tools_json"]) if row["tools_json"] else None,
        "source": row["source"],
        "group_id": row["group_id"],
        "status": row["status"],
        "content_hash": row["content_hash"],
        "token_count": row["token_count"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _payload(data: schemas.ExampleIn | schemas.ExamplePatch) -> tuple[list[dict], list[dict] | None]:
    messages = [m.model_dump() for m in (data.messages or [])]
    tools = list(data.tools) if data.tools else None
    return messages, tools


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------


def list_examples(
    conn: sqlite3.Connection,
    category: str | None = None,
    status: str | None = None,
    q: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[int, list[dict[str, Any]]]:
    where: list[str] = []
    params: list[Any] = []
    if category:
        where.append("category = ?")
        params.append(category)
    if status:
        where.append("status = ?")
        params.append(status)
    if q:
        where.append("(messages_json LIKE ? OR id LIKE ? OR COALESCE(group_id,'') LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like, like])
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    total = conn.execute(f"SELECT COUNT(*) AS n FROM examples {clause}", params).fetchone()["n"]
    rows = conn.execute(
        f"SELECT * FROM examples {clause} ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
        [*params, limit, offset],
    ).fetchall()
    return total, [row_to_out(r) for r in rows]


def get_row(conn: sqlite3.Connection, ex_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM examples WHERE id = ?", (ex_id,)).fetchone()


def all_framework(
    conn: sqlite3.Connection, exclude_id: str | None = None, status: str | None = None
) -> list[FwExample]:
    sql = "SELECT * FROM examples"
    params: list[Any] = []
    conds = []
    if exclude_id:
        conds.append("id != ?")
        params.append(exclude_id)
    if status:
        conds.append("status = ?")
        params.append(status)
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    return [row_to_framework(r) for r in conn.execute(sql, params).fetchall()]


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def token_count_for(fw_ex: FwExample, adapter_name: str | None = None) -> int | None:
    adapter = get_adapter(adapter_name)
    if not adapter.tokenizer_available():
        return None
    text = adapter.render_conversation(
        fw_ex.messages, tools=fw_ex.tools, add_generation_prompt=False
    )
    return adapter.count_tokens(text)


def validate_draft(
    conn: sqlite3.Connection,
    data: schemas.ExampleIn,
    adapter_name: str | None = None,
    exclude_id: str | None = None,
) -> Report:
    messages, tools = _payload(data)
    fw = FwExample(
        id=exclude_id or "draft",
        category=data.category,
        messages=messages,
        tools=tools,
        source=data.source,
        group_id=data.group_id,
        status="draft",
        content_hash=content_hash(messages, tools),
    )
    ctx = make_context(peers=all_framework(conn, exclude_id=exclude_id), adapter_name=adapter_name)
    return validate_example(fw, ctx)


# ---------------------------------------------------------------------------
# writes
# ---------------------------------------------------------------------------


def create_example(
    conn: sqlite3.Connection, data: schemas.ExampleIn, adapter_name: str | None = None
) -> tuple[dict[str, Any], Report]:
    messages, tools = _payload(data)
    h = content_hash(messages, tools)
    ex_id = ids.example_id()
    ts = now_iso()
    fw = FwExample(
        id=ex_id,
        category=data.category,
        messages=messages,
        tools=tools,
        source=data.source,
        group_id=data.group_id,
        status="draft",
        content_hash=h,
    )
    tokens = token_count_for(fw, adapter_name)
    conn.execute(
        """INSERT INTO examples
           (id, category, messages_json, tools_json, source, group_id, status,
            content_hash, token_count, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (
            ex_id,
            data.category,
            json.dumps(messages, ensure_ascii=False),
            json.dumps(tools, ensure_ascii=False) if tools else None,
            data.source,
            data.group_id,
            "draft",
            h,
            tokens,
            ts,
            ts,
        ),
    )
    conn.commit()
    report = validate_draft(conn, data, adapter_name, exclude_id=ex_id)
    return row_to_out(get_row(conn, ex_id)), report


def update_example(
    conn: sqlite3.Connection,
    ex_id: str,
    patch: schemas.ExamplePatch,
    adapter_name: str | None = None,
) -> tuple[dict[str, Any] | None, Report | None]:
    row = get_row(conn, ex_id)
    if row is None:
        return None, None
    current = row_to_out(row)
    merged = schemas.ExampleIn(
        category=patch.category or current["category"],
        messages=patch.messages if patch.messages is not None else current["messages"],
        tools=patch.tools if patch.tools is not None else current["tools"],
        source=patch.source or current["source"],
        group_id=patch.group_id if patch.group_id is not None else current["group_id"],
    )
    messages, tools = _payload(merged)
    h = content_hash(messages, tools)
    fw = FwExample(
        id=ex_id,
        category=merged.category,
        messages=messages,
        tools=tools,
        source=merged.source,
        group_id=merged.group_id,
        status=current["status"],
        content_hash=h,
    )
    tokens = token_count_for(fw, adapter_name)
    conn.execute(
        """UPDATE examples SET category=?, messages_json=?, tools_json=?, source=?,
           group_id=?, content_hash=?, token_count=?, updated_at=? WHERE id=?""",
        (
            merged.category,
            json.dumps(messages, ensure_ascii=False),
            json.dumps(tools, ensure_ascii=False) if tools else None,
            merged.source,
            merged.group_id,
            h,
            tokens,
            now_iso(),
            ex_id,
        ),
    )
    conn.commit()
    report = validate_draft(conn, merged, adapter_name, exclude_id=ex_id)
    return row_to_out(get_row(conn, ex_id)), report


def set_status(conn: sqlite3.Connection, ex_id: str, status: str) -> dict[str, Any] | None:
    if get_row(conn, ex_id) is None:
        return None
    conn.execute(
        "UPDATE examples SET status=?, updated_at=? WHERE id=?", (status, now_iso(), ex_id)
    )
    conn.commit()
    return row_to_out(get_row(conn, ex_id))


def delete_example(conn: sqlite3.Connection, ex_id: str) -> bool:
    cur = conn.execute("DELETE FROM examples WHERE id = ?", (ex_id,))
    conn.commit()
    return cur.rowcount > 0
