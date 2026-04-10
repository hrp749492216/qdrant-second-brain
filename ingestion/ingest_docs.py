#!/usr/bin/env python3
# ~/qdrant/ingestion/ingest_docs.py
"""
Ingest local knowledge directories into Qdrant.

Usage:
  python ingestion/ingest_docs.py
  python ingestion/ingest_docs.py --sync       # also delete removed files
  python ingestion/ingest_docs.py --dry-run    # show what would change
"""
import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from backend.db import (get_connection, init_tables, get_or_create_source,
                        upsert_chunk, delete_source, now_iso)
from ingestion.chunker import chunk_text, truncate_tokens
from ingestion.embedder import embed_texts
from ingestion.sparse import sparse_embed_batch

# ── Text extractors ────────────────────────────────────────────────────────────

def extract_pdf(path: str) -> str:
    from pypdf import PdfReader
    try:
        reader = PdfReader(path)
        return "\n".join(p.extract_text() or "" for p in reader.pages).strip()
    except Exception as e:
        return ""

def extract_docx(path: str) -> str:
    from docx import Document
    try:
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except Exception:
        return ""

def extract_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""

def extract_ipynb(path: str) -> str:
    import nbformat
    try:
        nb = nbformat.read(path, as_version=4)
        parts = []
        for cell in nb.cells:
            if cell.cell_type in ("markdown", "code"):
                parts.append(cell.source)
        return "\n\n".join(parts).strip()
    except Exception:
        return ""

EXTRACTORS = {
    ".pdf": extract_pdf, ".docx": extract_docx,
    ".txt": extract_text, ".md": extract_text, ".py": extract_text,
    ".ipynb": extract_ipynb,
}

def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# ── Main ───────────────────────────────────────────────────────────────────────

def collect_files() -> list[tuple[str, str]]:
    """Return list of (file_path, category) from KNOWLEDGE_DIRS."""
    results = []
    for dir_path, category in config.KNOWLEDGE_DIRS:
        if not os.path.isdir(dir_path):
            print(f"  [SKIP] {dir_path}")
            continue
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs
                       if d not in config.WATCHER_DENYLIST and not d.startswith(".")]
            for fname in files:
                ext = Path(fname).suffix.lower()
                if ext not in config.DOC_ALLOWLIST:
                    continue
                fpath = os.path.join(root, fname)
                if os.path.getsize(fpath) > config.MAX_FILE_SIZE_BYTES:
                    print(f"  [SKIP-SIZE] {fpath}")
                    continue
                results.append((fpath, category))
    return results

def ingest_file(fpath: str, category: str, conn, qdrant_client, dry_run: bool):
    ext = Path(fpath).suffix.lower()
    extractor = EXTRACTORS.get(ext)
    if not extractor:
        return
    text = extractor(fpath)
    if not text or len(text) < 20:
        conn.execute(
            "INSERT INTO ingestion_errors (source_path, error, occurred_at) VALUES (?,?,?)",
            [fpath, "no_extractable_text", now_iso()]
        )
        conn.commit()
        return

    # Rough character-based truncation (4 chars ~ 1 token on average)
    # before tokenizing to avoid tokenizer overflow (131K token limit)
    max_chars = 130000 * 4
    if len(text) > max_chars:
        text = text[:max_chars]

    # Fine-grained token truncation if needed
    text = truncate_tokens(text, max_tokens=130000)

    chunks = chunk_text(text)
    if not chunks:
        return

    title = Path(fpath).name
    fhash = file_hash(fpath)
    date  = "1970-01-01T00:00:00Z"
    try:
        mtime = os.path.getmtime(fpath)
        import datetime
        date = datetime.datetime.fromtimestamp(
            mtime, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        pass

    if dry_run:
        print(f"  [DRY] {fpath} → {len(chunks)} chunks")
        return

    source_id = get_or_create_source(conn, fpath, "document",
                                     platform="local", title=title,
                                     date=date, file_hash=fhash,
                                     category=category)

    # Delete stale tail chunks (file shrinkage)
    old_max = conn.execute(
        "SELECT MAX(chunk_index) FROM chunks WHERE source_id=?", [source_id]
    ).fetchone()[0]
    new_total = len(chunks)
    if old_max is not None and old_max >= new_total:
        stale_ids = [r["qdrant_point_id"] for r in conn.execute(
            "SELECT qdrant_point_id FROM chunks WHERE source_id=? AND chunk_index>=?",
            [source_id, new_total]
        ).fetchall()]
        if stale_ids:
            from qdrant_client.models import PointIdsList
            qdrant_client.delete(collection_name=config.QDRANT_COLLECTION,
                                 points_selector=PointIdsList(points=stale_ids))
        conn.execute("DELETE FROM chunks WHERE source_id=? AND chunk_index>=?",
                     [source_id, new_total])
        conn.commit()

    # Embed in batches
    child_texts = [c["child_text"] for c in chunks]
    for batch_start in range(0, len(child_texts), config.EMBED_BATCH_SIZE):
        batch_chunks = chunks[batch_start: batch_start + config.EMBED_BATCH_SIZE]
        batch_texts  = [c["child_text"] for c in batch_chunks]
        try:
            dense_vecs  = embed_texts(batch_texts)
            sparse_vecs = sparse_embed_batch(batch_texts)
        except Exception as exc:
            conn.execute(
                "INSERT INTO ingestion_errors (source_path, error, occurred_at) VALUES (?,?,?)",
                [fpath, str(exc), now_iso()]
            )
            conn.commit()
            continue

        for i, (chunk, dvec, svec) in enumerate(zip(batch_chunks, dense_vecs, sparse_vecs)):
            ci = batch_start + i
            upsert_chunk(conn, source_id, chunk["chunk_index"],
                         chunk["child_text"], chunk.get("parent_text") or "",
                         dvec, svec, qdrant_client)

        # Checkpoint
        conn.execute("""
            INSERT INTO ingestion_progress (file_path, last_chunk_index, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
              last_chunk_index=excluded.last_chunk_index,
              updated_at=excluded.updated_at
        """, [fpath, batch_start + len(batch_chunks) - 1, now_iso()])
        conn.commit()
        print(f"  [{batch_start+len(batch_chunks)}/{len(chunks)}] {Path(fpath).name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync",    action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force",   action="store_true")
    args = parser.parse_args()

    from qdrant_client import QdrantClient
    qdrant = QdrantClient(url=config.QDRANT_URL)
    conn   = get_connection()
    init_tables(conn)

    files = collect_files()
    print(f"Found {len(files)} files to process.")

    for fpath, category in files:
        print(f"Ingesting: {fpath}")
        try:
            ingest_file(fpath, category, conn, qdrant, args.dry_run)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            traceback.print_exc()

    if args.sync and not args.dry_run:
        disk_paths = {f for f, _ in files}
        db_paths   = {r["path"] for r in
                      conn.execute("SELECT path FROM sources WHERE source_type='document'").fetchall()}
        removed = db_paths - disk_paths
        total   = len(db_paths)
        if removed and not args.force:
            pct = len(removed) / max(total, 1)
            if pct > config.SYNC_MAX_DELETE_PCT:
                print(f"[SAFETY] Would delete {len(removed)}/{total} sources ({pct:.0%}). "
                      "Use --force to proceed.")
                sys.exit(1)
        for path in removed:
            src = conn.execute("SELECT id FROM sources WHERE path=?", [path]).fetchone()
            if src:
                delete_source(conn, src["id"], qdrant)
                print(f"  [DELETED] {path}")

        # Sweep Qdrant for orphaned vectors (point IDs not in SQLite chunks table)
        db_point_ids = {r["qdrant_point_id"] for r in
                        conn.execute("SELECT qdrant_point_id FROM chunks").fetchall()}
        orphan_ids = []
        next_offset = None
        while True:
            batch, next_offset = qdrant.scroll(
                collection_name=config.QDRANT_COLLECTION,
                limit=1000,
                offset=next_offset,
                with_payload=False,
                with_vectors=False,
            )
            for pt in batch:
                if pt.id not in db_point_ids:
                    orphan_ids.append(pt.id)
            if next_offset is None:
                break
        if orphan_ids:
            from qdrant_client.models import PointIdsList
            qdrant.delete(collection_name=config.QDRANT_COLLECTION,
                          points_selector=PointIdsList(points=orphan_ids))
            print(f"  [ORPHAN] Deleted {len(orphan_ids)} orphaned Qdrant vectors.")

    print("Ingestion complete.")

if __name__ == "__main__":
    main()
