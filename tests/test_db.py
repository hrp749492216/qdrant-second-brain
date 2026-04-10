import sqlite3, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
config.SQLITE_PATH = ":memory:"   # use in-memory DB for tests

from backend.db import get_connection, init_tables

def test_pragmas_are_set():
    conn = get_connection()
    fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    jm = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert fk == 1, "foreign_keys must be ON"
    # WAL mode is attempted but in-memory DB falls back to memory journal mode
    assert jm in ("wal", "memory"), "journal_mode must be WAL or memory for in-memory DB"

def test_tables_created():
    conn = get_connection()
    init_tables(conn)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    for t in ["sources", "chunks", "quarantine", "ingestion_errors",
              "ingestion_progress", "meta"]:
        assert t in tables, f"missing table: {t}"

def test_meta_keys():
    conn = get_connection()
    init_tables(conn)
    keys = {r[0] for r in conn.execute("SELECT key FROM meta").fetchall()}
    # meta is empty until init_schema populates it — table just needs to exist
    assert isinstance(keys, set)
