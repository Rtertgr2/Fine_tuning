"""SQLite access layer."""
# Schema version bumped to 3 for training_runs table

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

CREATE TABLE IF NOT EXISTS model_versions (
    version          TEXT PRIMARY KEY,
    dataset_version  TEXT,
    train_run        TEXT,
    llama_cpp_commit TEXT,
    quant            TEXT NOT NULL
                     CHECK (quant IN ('Q4_K_M','Q5_K_M','Q8_0','Q2_K','Q3_K_M','Q6_K')),
    gguf_sha256      TEXT,
    eval_report      TEXT,
    status           TEXT NOT NULL DEFAULT 'candidate'
                     CHECK (status IN ('candidate','staging','production','retired')),
    manifest_json    TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    promoted_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_model_status ON model_versions(status);

-- Phase 3: Training runs
CREATE TABLE IF NOT EXISTS training_runs (
    run_id        TEXT PRIMARY KEY,
    config_json   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending','running','completed','stopped','failed')),
    dataset_version TEXT,
    adapter_path  TEXT,
    metrics_path  TEXT,
    report_path   TEXT,
    preflight_json TEXT,
    created_at    TEXT NOT NULL,
    completed_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_training_status ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_training_dataset ON training_runs(dataset_version);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

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

-- ─── Phase 6: Active Learning ───────────────────────────────────────────

CREATE TABLE IF NOT EXISTS activity_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT NOT NULL,
    ts               TEXT NOT NULL,
    model_version    TEXT NOT NULL,
    state            TEXT NOT NULL CHECK (state IN ('plan','code','review','idle')),
    event            TEXT NOT NULL CHECK (event IN (
                        'message','tool_call','tool_result','circuit_breaker',
                        'verdict','override','outcome','cancel','undo'
                     )),
    data             TEXT NOT NULL,  -- JSON, secrets redacted at write time
    project          TEXT,
    redaction_count  INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_al_session  ON activity_log(session_id);
CREATE INDEX IF NOT EXISTS idx_al_model     ON activity_log(model_version);
CREATE INDEX IF NOT EXISTS idx_al_event     ON activity_log(event);
CREATE INDEX IF NOT EXISTS idx_al_state     ON activity_log(state);
CREATE INDEX IF NOT EXISTS idx_al_ts        ON activity_log(ts);

CREATE TABLE IF NOT EXISTS case_queue (
    case_id           TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL,
    model_version     TEXT NOT NULL,
    case_type         TEXT NOT NULL CHECK (case_type IN (
                          'loop','syntax','false_negative','false_positive',
                          'cancel','success','sec'
                      )),
    priority          INTEGER NOT NULL DEFAULT 1 CHECK (priority BETWEEN 1 AND 5),
    signal            TEXT NOT NULL,
    events_json       TEXT NOT NULL,  -- Full session events for review
    project           TEXT,
    status            TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN (
                            'pending','editing','approved','rejected','merged'
                        )),
    reviewer          TEXT,
    second_reviewer   TEXT,  -- Required for sec category before approval
    rejection_reason  TEXT,
    edited_example_json TEXT,  -- Final approved example (JSON)
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cq_status   ON case_queue(status);
CREATE INDEX IF NOT EXISTS idx_cq_type     ON case_queue(case_type);
CREATE INDEX IF NOT EXISTS idx_cq_priority ON case_queue(priority);
CREATE INDEX IF NOT EXISTS idx_cq_model    ON case_queue(model_version);
CREATE INDEX IF NOT EXISTS idx_cq_session  ON case_queue(session_id);
"""

SCHEMA_VERSION = "4"


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
