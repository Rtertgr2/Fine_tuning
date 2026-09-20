"""Stats and adapter info endpoints."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends

from backend.adapters.registry import get_adapter, list_adapters
from backend.app import config
from backend.app.db import get_db
from backend.app.services import stats as stats_service

router = APIRouter(tags=["stats"])


@router.get("/stats")
def stats(conn: sqlite3.Connection = Depends(get_db)):
    return stats_service.compute_stats(conn)


@router.get("/adapters")
def adapters():
    active = get_adapter()
    return {
        "active": active.name,
        "available": list_adapters(),
        "max_seq_len": config.MAX_SEQ_LEN,
        "tokenizer_available": active.tokenizer_available(),
    }