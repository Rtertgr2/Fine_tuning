"""Draft preview endpoints for the editor UI (T1.5).

Read-only: they take a draft in the exact payload shape of POST /examples and
show what the training-time chat template produces plus the real token count.
The UI must not reimplement any validator rule — validation lives in
POST /validate, rendering lives here (both backend, one source of truth).
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends

from backend.adapters.registry import get_adapter
from backend.app import config, schemas
from backend.app.db import get_db
from backend.app.services import examples as svc

router = APIRouter(tags=["draft"])


@router.post("/render")
def render_draft(data: schemas.ExampleIn, conn: sqlite3.Connection = Depends(get_db)):
    """Render a draft with the active chat template and count real tokens.

    Applies the same auto tool-definition injection as saving does, so the
    preview is byte-identical to what would be trained on.
    """
    adapter = get_adapter()
    messages, tools = svc._payload(data)
    rendered = adapter.render_conversation(
        messages, tools=tools, add_generation_prompt=False
    )
    available = adapter.tokenizer_available()
    token_count = adapter.count_tokens(rendered) if available else None
    return {
        "rendered": rendered,
        "token_count": token_count,
        "adapter": adapter.name,
        "template_kind": adapter.pick_template(tools),
        "tools_auto_injected": bool(tools) and data.tools is None,
        "max_seq_len": config.MAX_SEQ_LEN,
        "tokenizer_available": available,
    }


@router.get("/tools")
def tool_registry():
    """Tool definitions for the editor's `tools` prefill (single source: T2/T3)."""
    from backend.tools.registry import tool_names, tool_schemas

    return {"names": tool_names(), "schemas": tool_schemas()}
