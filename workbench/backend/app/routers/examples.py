"""Example endpoints (plan 01 §7)."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse

from backend.app import schemas
from backend.app.db import get_db
from backend.app.services import examples as svc
from backend.app.services import io_jsonl
from backend.validators.registry import report_to_dict

router = APIRouter(tags=["examples"])


@router.get("/examples", response_model=schemas.ExampleListOut)
def list_examples(
    category: schemas.Category | None = None,
    status: schemas.Status | None = None,
    q: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    conn: sqlite3.Connection = Depends(get_db),
):
    total, items = svc.list_examples(conn, category, status, q, limit, offset)
    return {"total": total, "items": items}


@router.post("/examples", status_code=201, response_model=schemas.ExampleWithReport)
def create_example(data: schemas.ExampleIn, conn: sqlite3.Connection = Depends(get_db)):
    record, report = svc.create_example(conn, data)
    return {"example": record, "report": report_to_dict(report)}


@router.get("/examples/{ex_id}", response_model=schemas.ExampleOut)
def get_example(ex_id: str, conn: sqlite3.Connection = Depends(get_db)):
    row = svc.get_row(conn, ex_id)
    if row is None:
        raise HTTPException(404, "example not found")
    return svc.row_to_out(row)


@router.put("/examples/{ex_id}", response_model=schemas.ExampleWithReport)
def update_example(
    ex_id: str, patch: schemas.ExamplePatch, conn: sqlite3.Connection = Depends(get_db)
):
    record, report = svc.update_example(conn, ex_id, patch)
    if record is None:
        raise HTTPException(404, "example not found")
    return {"example": record, "report": report_to_dict(report)}


@router.post("/examples/{ex_id}/status", response_model=schemas.ExampleOut)
def change_status(
    ex_id: str, change: schemas.StatusChange, conn: sqlite3.Connection = Depends(get_db)
):
    record = svc.set_status(conn, ex_id, change.status)
    if record is None:
        raise HTTPException(404, "example not found")
    return record


@router.delete("/examples/{ex_id}", status_code=204)
def delete_example(ex_id: str, conn: sqlite3.Connection = Depends(get_db)):
    if not svc.delete_example(conn, ex_id):
        raise HTTPException(404, "example not found")


@router.post("/import", response_model=schemas.ImportResult)
def import_jsonl(body: bytes = Body(media_type="text/plain"), conn: sqlite3.Connection = Depends(get_db)):
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(400, f"body must be UTF-8 JSONL: {exc}") from exc
    return io_jsonl.import_lines(conn, text)


@router.get("/export", response_class=PlainTextResponse)
def export_jsonl(
    category: schemas.Category | None = None,
    status: schemas.Status | None = None,
    conn: sqlite3.Connection = Depends(get_db),
):
    _total, items = svc.list_examples(conn, category, status, None, limit=100000)
    rows = [svc.get_row(conn, item["id"]) for item in items]
    return io_jsonl.export_rows([r for r in rows if r is not None])
