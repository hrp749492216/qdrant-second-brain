#!/usr/bin/env python3
# ~/qdrant/ingestion/ingest_convos.py
"""
Ingest AI conversation export files (Claude, ChatGPT, Gemini).

Usage:
  python ingestion/ingest_convos.py ~/Downloads/conversations.json
"""
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from backend.db import (get_connection, init_tables, get_or_create_source,
                        upsert_chunk, now_iso)
from ingestion.chunker import chunk_text, count_tokens, truncate_tokens
from ingestion.embedder import embed_texts
from ingestion.sparse import sparse_embed_batch
from ingestion.parsers.base import detect_platform

def get_parser(platform: str):
    if platform == "claude":
        from ingestion.parsers.claude import ClaudeParser
        return ClaudeParser()
    elif platform == "chatgpt":
        from ingestion.parsers.chatgpt import ChatGPTParser
        return ChatGPTParser()
    elif platform == "gemini":
        from ingestion.parsers.gemini import GeminiParser
        return GeminiParser()
    raise ValueError(f"Unknown platform: {platform}")

def chunk_conversation(conv) -> list[dict]:
    """Chunk a Conversation into message-group chunks (1-3 turns)."""
    turns = []
    msgs = conv.messages
    i = 0
    while i < len(msgs):
        # Collect 1 user + 1 assistant = 1 turn
        turn_texts = []
        while i < len(msgs) and len(turn_texts) < 2:
            m = msgs[i]
            turn_texts.append(f"{m.role.upper()}: {m.content}")
            i += 1
        turns.append("\n\n".join(turn_texts))

    # Group turns into 1-3 per chunk, sub-chunk oversized ones
    chunks = []
    idx = 0
    group = []
    MAX_TOKENS = config.CHUNK_SIZE_TOKENS

    for turn in turns:
        if count_tokens(turn) > MAX_TOKENS:
            # Sub-chunk at paragraph boundaries
            paragraphs = turn.split("\n\n")
            buf = ""
            for para in paragraphs:
                candidate = (buf + "\n\n" + para).strip()
                if count_tokens(candidate) > MAX_TOKENS and buf:
                    group.append(buf)
                    buf = para
                else:
                    buf = candidate
            if buf:
                group.append(buf)
        else:
            group.append(turn)

        if len(group) >= 3:
            chunks.append({
                "chunk_index": idx,
                "child_text":  "\n\n".join(group),
                "parent_text": None,
            })
            idx += 1
            group = []

    if group:
        chunks.append({"chunk_index": idx, "child_text": "\n\n".join(group), "parent_text": None})

    return chunks

def ingest_file(file_path: str, conn, qdrant_client):
    data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    platform = detect_platform(data)
    if not platform:
        conn.execute(
            "INSERT INTO quarantine (path, detected_at, status, reason) VALUES (?,?,?,?)",
            [file_path, now_iso(), "rejected", "unknown_format"]
        )
        conn.commit()
        print(f"[REJECTED] Unknown format: {file_path}")
        return

    parser = get_parser(platform)
    conversations = parser.parse(Path(file_path))
    print(f"Parsed {len(conversations)} conversations from {Path(file_path).name} ({platform})")

    # Checkpoint: resume from last_convo_index
    progress = conn.execute(
        "SELECT last_convo_index FROM ingestion_progress WHERE file_path=?", [file_path]
    ).fetchone()
    resume_from = (progress["last_convo_index"] + 1) if progress else 0

    for convo_idx, conv in enumerate(conversations):
        if convo_idx < resume_from:
            continue
        source_path = f"{platform}::{conv.conversation_id}"
        date_rfc = f"{conv.date}T00:00:00Z" if "T" not in conv.date else conv.date
        source_id = get_or_create_source(
            conn, source_path, "conversation",
            platform=platform, title=conv.title, date=date_rfc,
        )

        chunks = chunk_conversation(conv)
        if not chunks:
            continue

        child_texts = [c["child_text"] for c in chunks]
        for batch_start in range(0, len(child_texts), config.EMBED_BATCH_SIZE):
            batch = chunks[batch_start: batch_start + config.EMBED_BATCH_SIZE]
            btexts = [c["child_text"] for c in batch]
            try:
                dense_vecs  = embed_texts(btexts)
                sparse_vecs = sparse_embed_batch(btexts)
            except Exception as exc:
                conn.execute(
                    "INSERT INTO ingestion_errors (source_path, error, occurred_at) VALUES (?,?,?)",
                    [source_path, str(exc), now_iso()]
                )
                conn.commit()
                continue
            for chunk, dvec, svec in zip(batch, dense_vecs, sparse_vecs):
                upsert_chunk(conn, source_id, chunk["chunk_index"],
                             chunk["child_text"], "",
                             dvec, svec, qdrant_client)

        # Checkpoint after each conversation
        conn.execute("""
            INSERT INTO ingestion_progress (file_path, last_chunk_index, last_convo_index, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
              last_convo_index=excluded.last_convo_index,
              last_chunk_index=excluded.last_chunk_index,
              updated_at=excluded.updated_at
        """, [file_path, len(chunks) - 1, convo_idx, now_iso()])
        conn.commit()
        print(f"  [{convo_idx+1}/{len(conversations)}] {conv.title[:60]}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python ingestion/ingest_convos.py <export_file> [...]")
        sys.exit(1)

    from qdrant_client import QdrantClient
    qdrant = QdrantClient(url=config.QDRANT_URL)
    conn   = get_connection()
    init_tables(conn)

    for file_path in sys.argv[1:]:
        print(f"\nProcessing: {file_path}")
        try:
            ingest_file(file_path, conn, qdrant)
        except Exception as exc:
            print(f"[ERROR] {exc}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
