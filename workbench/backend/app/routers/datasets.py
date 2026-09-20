"""Dataset version endpoints (T1.8) — build, list, export."""

from __future__ import annotations

import io
import sqlite3
import zipfile

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response

from backend.app import schemas
from backend.app.db import get_db
from backend.app.services import datasets as ds_service

router = APIRouter(tags=["datasets"])


@router.post("/datasets", status_code=201, response_model=schemas.DatasetOut)
def create_dataset(payload: schemas.DatasetCreateIn, conn: sqlite3.Connection = Depends(get_db)):
    try:
        record = ds_service.build_dataset(
            conn,
            name=payload.name,
            seed=payload.seed,
            val_ratio=payload.val_ratio,
            categories=list(payload.categories) if payload.categories else None,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return record


@router.get("/datasets", response_model=list[schemas.DatasetOut])
def list_datasets(conn: sqlite3.Connection = Depends(get_db)):
    return ds_service.list_datasets(conn)


@router.get("/datasets/{ds_id}", response_model=schemas.DatasetOut)
def get_dataset(ds_id: str, conn: sqlite3.Connection = Depends(get_db)):
    record = ds_service.get_dataset(conn, ds_id)
    if record is None:
        raise HTTPException(404, "dataset not found")
    return record


@router.get("/datasets/{ds_id}/export")
def export_dataset(
    ds_id: str,
    which: str = Query(default="zip", pattern="^(train|val|zip)$"),
    conn: sqlite3.Connection = Depends(get_db),
):
    record = ds_service.get_dataset(conn, ds_id)
    if record is None:
        raise HTTPException(404, "dataset not found")
    base = ds_service.dataset_dir(ds_id)

    if which in ("train", "val"):
        path = base / f"{which}.jsonl"
        if not path.exists():
            raise HTTPException(404, f"{which}.jsonl missing on disk")
        return FileResponse(path, media_type="application/x-ndjson", filename=f"{ds_id}-{which}.jsonl")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in ("train.jsonl", "val.jsonl", "manifest.json"):
            path = base / filename
            if path.exists():
                zf.write(path, arcname=f"{ds_id}/{filename}")
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{ds_id}.zip"'},
    )