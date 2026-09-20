"""SQLite access layer."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS examples (
    id            TEXT PRIMARY KEY,
    category      TEXT NOT NULL CHECK (category IN ('tool','loop','plan','sec')),
    messages_json TEXT NOT NULL,
    tools_json    TEXT,
    source        TEXT NOT NULL DEFAULT 'manual'
                  CHECK (source IN ('manual','generated','active_learning')),
    group_id      TEXT,
    status        TEXT NOT NULL DEFAULT 'draft'
                  CHECK (status IN ('draft','approved','rejected')),
    content_hash  TEXT NOT NULL,
    token_count   INTEGER,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_examples_category ON examples(category);
CREATE INDEX IF NOT EXISTS idx_examples_status   ON examples(status);
CREATE INDEX IF NOT EXISTS idx_examples_group    ON examples(group_id);
CREATE INDEX IF NOT EXISTS idx_examples_hash     ON examples(content_hash);

CREATE TABLE IF NOT EXISTS dataset_versions (
    id               TEXT PRIMARY KEY,
    name             TEXT,
    example_ids_json TEXT NOT NULL,
    split_json       TEXT NOT NULL,
    seed             INTEGER NOT NULL,
    manifest_json    TEXT NOT NULL,
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

SCHEMA_VERSION = "1"


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str | Path | None = None) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        conn.commit()
    finally:
        conn.close()


def get_db():
    """FastAPI dependency: one connection per request."""
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()
