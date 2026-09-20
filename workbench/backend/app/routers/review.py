"""Review queue API endpoints (T2.9)."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.db import get_db
from backend.app.services import review as svc

router = APIRouter(prefix="/review", tags=["review"])


@router.get("/queue")
def list_pending_reviews(
    category: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    conn: sqlite3.Connection = Depends(get_db),
):
    """List pending review items."""
    total, items = svc.list_pending(conn, category, limit, offset)
    return {"total": total, "items": items}


@router.get("/queue/{item_id}")
def get_review_item(
    item_id: str,
    conn: sqlite3.Connection = Depends(get_db),
):
    """Get a single review item."""
    item = svc.get_item(conn, item_id)
    if item is None:
        raise HTTPException(404, "review item not found")
    return item


@router.post("/queue/{item_id}/approve")
def approve_item(
    item_id: str,
    reviewer: str | None = None,
    conn: sqlite3.Connection = Depends(get_db),
):
    """Approve a review item: moves it to the examples table."""
    result = svc.approve(conn, item_id, reviewer)
    if result is None:
        raise HTTPException(404, "review item not found")
    return result


@router.post("/queue/{item_id}/reject")
def reject_item(
    item_id: str,
    reason: str,
    reviewer: str | None = None,
    conn: sqlite3.Connection = Depends(get_db),
):
    """Reject a review item with a reason."""
    result = svc.reject(conn, item_id, reason, reviewer)
    if result is None:
        raise HTTPException(404, "review item not found")
    return result


@router.post("/queue/{item_id}/edit")
def edit_and_approve(
    item_id: str,
    edited_data: dict,
    reviewer: str | None = None,
    conn: sqlite3.Connection = Depends(get_db),
):
    """Edit example data and approve."""
    result = svc.edit_and_approve(conn, item_id, edited_data, reviewer)
    if result is None:
        raise HTTPException(404, "review item not found")
    return result
