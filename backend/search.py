# ~/qdrant/backend/search.py
import threading
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, Prefetch, FusionQuery, Fusion
import config
from ingestion.embedder import embed_texts
from ingestion.sparse import sparse_embed
from ingestion.chunker import truncate_tokens

# ── Singleton clients ──────────────────────────────────────────────────────────
_qdrant: Optional[QdrantClient] = None
_reranker = None
_reranker_lock = threading.Lock()  # threading.Lock (not asyncio) — acquired from thread pool

def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=config.QDRANT_URL)
    return _qdrant

def get_reranker():
    global _reranker
    if _reranker is None:
        # FlagEmbedding.FlagReranker broken on Python 3.14 (is_torch_fx_available removed).
        # sentence-transformers CrossEncoder provides identical bge-reranker-v2-m3 support.
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(config.RERANKER_MODEL, device=config.RERANKER_DEVICE)
    return _reranker

def warmup_reranker():
    r = get_reranker()
    r.predict([["warmup", "warmup"]], apply_softmax=True)

# ── Filter builder ─────────────────────────────────────────────────────────────

def build_filter(source_type=None, platform=None, category=None,
                 date_from=None, date_to=None) -> Optional[Filter]:
    conditions = []
    if source_type:
        conditions.append(FieldCondition(key="source_type", match=MatchValue(value=source_type)))
    if platform:
        conditions.append(FieldCondition(key="platform", match=MatchValue(value=platform)))
    if category:
        conditions.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if date_from or date_to:
        conditions.append(FieldCondition(
            key="date",
            range=Range(gte=date_from, lte=date_to),
        ))
    return Filter(must=conditions) if conditions else None

# ── Search ─────────────────────────────────────────────────────────────────────

def search(
    query: str,
    conn,
    source_type: Optional[str] = None,
    platform: Optional[str] = None,
    category: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = config.SEARCH_TOP_K,
    offset: int = 0,
) -> list[dict]:
    qdrant = get_qdrant()

    # Embed query
    dense_vec  = embed_texts([query], instruction=config.QUERY_INSTRUCTION)[0]
    sparse_vec = sparse_embed(query)

    # Build filter — pushed into BOTH Prefetch objects
    active_filter = build_filter(source_type, platform, category, date_from, date_to)

    # Hybrid search: server-side RRF
    hits = qdrant.query_points(
        collection_name=config.QDRANT_COLLECTION,
        prefetch=[
            Prefetch(query=dense_vec,  using="dense",  limit=config.RERANK_CANDIDATES,
                     filter=active_filter),
            Prefetch(query=sparse_vec, using="sparse", limit=config.RERANK_CANDIDATES,
                     filter=active_filter),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=config.RERANK_CANDIDATES,
        with_payload=True,
    ).points

    if not hits:
        return []

    # Fetch chunk_text from SQLite
    sqlite_ids = [h.payload["sqlite_chunk_id"] for h in hits]
    placeholders = ",".join("?" * len(sqlite_ids))
    rows = {r["id"]: r for r in conn.execute(
        f"SELECT id, chunk_text, parent_text FROM chunks WHERE id IN ({placeholders})",
        sqlite_ids
    ).fetchall()}

    candidates = []
    for hit in hits:
        sid = hit.payload.get("sqlite_chunk_id")
        row = rows.get(sid)
        if not row:
            continue
        candidates.append({
            "score":       hit.score,
            "sqlite_id":   sid,
            "chunk_text":  row["chunk_text"],
            "parent_text": row["parent_text"] or "",
            "payload":     hit.payload,
        })

    if not candidates:
        return []

    # Rerank (cross-encoder, truncate to 400 tokens)
    # Lock here — bge-reranker is not thread-safe; acquired from thread-pool context
    reranker = get_reranker()
    pairs = [
        [query, truncate_tokens(c["chunk_text"], config.RERANK_MAX_CHUNK_TOKENS)]
        for c in candidates
    ]
    with _reranker_lock:
        scores = reranker.predict(pairs, apply_softmax=True, batch_size=config.RERANK_BATCH_SIZE)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    top = candidates[offset: offset + limit]

    return [
        {
            "score":       c["rerank_score"],
            "title":       c["payload"].get("title", ""),
            "platform":    c["payload"].get("platform", ""),
            "category":    c["payload"].get("category", ""),
            "date":        c["payload"].get("date", ""),
            "source_path": c["payload"].get("source_path", ""),
            "chunk_text":  c["chunk_text"],
            "parent_text": c["parent_text"],
        }
        for c in top
    ]
