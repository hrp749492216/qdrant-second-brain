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


import uuid as _uuid
import hashlib
import datetime


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_normalize(text: str) -> str:
    """Whitespace-normalize (no lowercase) and SHA-256 hash."""
    import re
    normalized = re.sub(r'\s+', ' ', text.strip())
    return hashlib.sha256(normalized.encode()).hexdigest()


def make_point_id(source_id: int, chunk_index: int) -> str:
    return str(_uuid.uuid5(config.NAMESPACE_BRAIN, f"{source_id}:{chunk_index}"))


def get_or_create_source(conn, path: str, source_type: str,
                          platform: str = "local", title: str = "",
                          date: str = "1970-01-01T00:00:00Z",
                          file_hash: str = "",
                          category: str = "") -> int:
    row = conn.execute("SELECT id FROM sources WHERE path = ?", [path]).fetchone()
    if row:
        return row["id"]
    conn.execute(
        "INSERT INTO sources (path, source_type, platform, title, date, ingested_at, file_hash, category) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [path, source_type, platform, title, date, now_iso(), file_hash, category],
    )
    conn.commit()
    return conn.execute("SELECT id FROM sources WHERE path = ?", [path]).fetchone()["id"]


def upsert_chunk(conn, source_id: int, chunk_index: int,
                 chunk_text: str, parent_text: str,
                 dense_vec: list, sparse_vec,
                 qdrant_client) -> None:
    """Two-phase write: Qdrant upsert first (idempotent), then SQLite commit."""
    from qdrant_client.models import PointStruct
    content_hash = sha256_normalize(chunk_text)
    point_id     = make_point_id(source_id, chunk_index)

    # Check if unchanged
    existing = conn.execute(
        "SELECT content_hash FROM chunks WHERE source_id=? AND chunk_index=?",
        [source_id, chunk_index]
    ).fetchone()
    if existing and existing["content_hash"] == content_hash:
        return  # no change

    # Get source metadata for payload (read before any write)
    src = conn.execute("SELECT * FROM sources WHERE id=?", [source_id]).fetchone()

    # Determine the sqlite_chunk_id for existing rows (new rows get id after insert)
    existing_id_row = conn.execute(
        "SELECT id FROM chunks WHERE source_id=? AND chunk_index=?",
        [source_id, chunk_index]
    ).fetchone()

    # 1. Qdrant upsert first (idempotent — can retry on failure)
    # sqlite_chunk_id is set to existing id if available; updated after SQLite write if new
    sqlite_chunk_id_placeholder = existing_id_row["id"] if existing_id_row else None
    qdrant_client.upsert(
        collection_name=config.QDRANT_COLLECTION,
        points=[PointStruct(
            id=point_id,
            vector={"dense": dense_vec, "sparse": sparse_vec},
            payload={
                "source_type":     src["source_type"],
                "platform":        src["platform"],
                "category":        src["category"],
                "title":           src["title"],
                "date":            src["date"],
                "source_path":     src["path"],
                "chunk_index":     chunk_index,
                "content_hash":    content_hash,
                "source_id":       source_id,
                "sqlite_chunk_id": sqlite_chunk_id_placeholder,
            }
        )],
        wait=True,
    )

    # 2. SQLite commit only after Qdrant succeeds
    with conn:
        conn.execute("""
            INSERT INTO chunks
              (source_id, qdrant_point_id, chunk_index, chunk_text, parent_text,
               content_hash, embedding_model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, chunk_index) DO UPDATE SET
              chunk_text=excluded.chunk_text,
              parent_text=excluded.parent_text,
              content_hash=excluded.content_hash,
              embedding_model_version=excluded.embedding_model_version,
              qdrant_point_id=excluded.qdrant_point_id
        """, [source_id, point_id, chunk_index, chunk_text, parent_text,
              content_hash, config.EMBEDDING_MODEL])

        # If this was a new row, update Qdrant payload with the real sqlite_chunk_id
        if sqlite_chunk_id_placeholder is None:
            sqlite_chunk_id = conn.execute(
                "SELECT id FROM chunks WHERE source_id=? AND chunk_index=?",
                [source_id, chunk_index]
            ).fetchone()["id"]
            qdrant_client.set_payload(
                collection_name=config.QDRANT_COLLECTION,
                payload={"sqlite_chunk_id": sqlite_chunk_id},
                points=[point_id],
            )


def delete_source(conn, source_id: int, qdrant_client) -> None:
    """Delete all chunks for a source from Qdrant and SQLite."""
    from qdrant_client.models import PointIdsList
    point_ids = [r["qdrant_point_id"] for r in
                 conn.execute("SELECT qdrant_point_id FROM chunks WHERE source_id=?",
                              [source_id]).fetchall()]
    if point_ids:
        qdrant_client.delete(
            collection_name=config.QDRANT_COLLECTION,
            points_selector=PointIdsList(points=point_ids),
        )
    conn.execute("DELETE FROM sources WHERE id=?", [source_id])
    conn.commit()
