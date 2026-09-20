"""JSONL import/export (T1.6) — lossless round-trip.

Export line format (matches the plan-00 schema):
  {"id", "category", "messages", "tools", "meta": {"source", "group",
   "status", "content_hash", "token_count"}}
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable

from pydantic import ValidationError

from backend.adapters.base import content_hash
from backend.app import ids, schemas
from backend.app.services import examples as ex_service


def export_line(row: sqlite3.Row) -> str:
    out = ex_service.row_to_out(row)
    payload = {
        "id": out["id"],
        "category": out["category"],
        "messages": out["messages"],
        "tools": out["tools"],
        "meta": {
            "source": out["source"],
            "group": out["group_id"],
            "status": out["status"],
            "content_hash": out["content_hash"],
            "token_count": out["token_count"],
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def export_rows(rows: Iterable[sqlite3.Row]) -> str:
    lines = [export_line(r) for r in rows]
    return "\n".join(lines) + ("\n" if lines else "")


def parse_import_line(line: str) -> tuple[schemas.ExampleIn, str | None, str | None]:
    obj = json.loads(line)
    if not isinstance(obj, dict):
        raise ValueError("line must be a JSON object")
    meta = obj.get("meta") or {}
    status = meta.get("status")
    if status not in (None, "draft", "approved", "rejected"):
        raise ValueError(f"invalid meta.status {status!r}")
    data = schemas.ExampleIn(
        category=obj.get("category"),
        messages=obj.get("messages") or [],
        tools=obj.get("tools"),
        source=meta.get("source", "manual"),
        group_id=meta.get("group"),
    )
    ex_id = obj.get("id")
    if ex_id is not None and not isinstance(ex_id, str):
        raise ValueError("id must be a string")
    return data, status, ex_id


def import_lines(
    conn: sqlite3.Connection,
    text: str,
    adapter_name: str | None = None,
    default_status: str = "draft",
) -> dict[str, Any]:
    """Import JSONL text; returns {imported, failed, errors, example_ids}."""
    existing_hashes = {
        r["content_hash"] for r in conn.execute("SELECT content_hash FROM examples").fetchall()
    }
    imported = 0
    failed = 0
    errors: list[dict[str, Any]] = []
    example_ids: list[str] = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            data, status, wanted_id = parse_import_line(line)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            failed += 1
            errors.append({"line": lineno, "error": str(exc)[:300]})
            continue

        messages = [m.model_dump() for m in data.messages]
        h = content_hash(messages, data.tools)
        if h in existing_hashes:
            failed += 1
            errors.append({"line": lineno, "error": "duplicate of an existing example"})
            continue

        ex_id = wanted_id or ids.example_id()
        if conn.execute("SELECT 1 FROM examples WHERE id = ?", (ex_id,)).fetchone():
            ex_id = ids.example_id()  # keep the id when free, else mint a new one
        ts = ex_service.now_iso()
        conn.execute(
            """INSERT INTO examples
               (id, category, messages_json, tools_json, source, group_id, status,
                content_hash, token_count, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ex_id,
                data.category,
                json.dumps(messages, ensure_ascii=False),
                json.dumps(data.tools, ensure_ascii=False) if data.tools else None,
                data.source,
                data.group_id,
                status or default_status,
                h,
                None,
                ts,
                ts,
            ),
        )
        existing_hashes.add(h)
        example_ids.append(ex_id)
        imported += 1

    conn.commit()
    return {
        "imported": imported,
        "failed": failed,
        "errors": errors,
        "example_ids": example_ids,
    }
