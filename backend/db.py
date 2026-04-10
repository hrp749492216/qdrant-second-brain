# ~/qdrant/backend/db.py
import sqlite3
from typing import Optional
import config

_DDL = """
CREATE TABLE IF NOT EXISTS sources (
    id           INTEGER PRIMARY KEY,
    path         TEXT UNIQUE NOT NULL,
    source_type  TEXT NOT NULL,
    platform     TEXT,
    title        TEXT,
    date         TEXT,
    ingested_at  TEXT NOT NULL,
    file_hash    TEXT,
    category     TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS chunks (
    id                      INTEGER PRIMARY KEY,
    source_id               INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    qdrant_point_id         TEXT NOT NULL UNIQUE,
    chunk_index             INTEGER NOT NULL,
    chunk_text              TEXT NOT NULL,
    parent_text             TEXT,
    content_hash            TEXT NOT NULL,
    embedding_model_version TEXT NOT NULL,
    UNIQUE(source_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS quarantine (
    id             INTEGER PRIMARY KEY,
    path           TEXT NOT NULL,
    detected_at    TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'pending',
    ingested_at    TEXT,
    ingested_count INTEGER,
    reason         TEXT
);

CREATE TABLE IF NOT EXISTS ingestion_errors (
    id          INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL,
    error       TEXT NOT NULL,
    occurred_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingestion_progress (
    file_path         TEXT PRIMARY KEY,
    last_chunk_index  INTEGER NOT NULL,
    last_convo_index  INTEGER,
    updated_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_source_id       ON chunks(source_id);
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash    ON chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_qdrant_point_id ON chunks(qdrant_point_id);
CREATE INDEX IF NOT EXISTS idx_chunks_emb_version     ON chunks(embedding_model_version);
CREATE INDEX IF NOT EXISTS idx_sources_path           ON sources(path);
CREATE INDEX IF NOT EXISTS idx_quarantine_status      ON quarantine(status);
"""

def get_connection(path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or config.SQLITE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn

def init_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(_DDL)
    conn.commit()
