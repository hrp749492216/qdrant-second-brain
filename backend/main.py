# ~/qdrant/backend/main.py
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import asyncio

from fastapi import FastAPI, Query, BackgroundTasks, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

import config
from backend.db import get_connection, init_tables, now_iso
from backend.search import search as _search, warmup_reranker

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warmup reranker on startup (hides JIT latency from first real query)
    print("[main] Warming up reranker...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, warmup_reranker)
    print("[main] Reranker ready.")
    yield

app = FastAPI(title="Second Brain", lifespan=lifespan)

# ── Static frontend ───────────────────────────────────────────────────────────
_frontend = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend.exists():
    app.mount("/app", StaticFiles(directory=str(_frontend), html=True), name="frontend")

# ── DB dependency ─────────────────────────────────────────────────────────────

def get_db():
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/search")
async def search_endpoint(
    q:           str            = Query(..., min_length=1),
    source_type: Optional[str]  = Query(None),
    platform:    Optional[str]  = Query(None),
    category:    Optional[str]  = Query(None),
    date_from:   Optional[str]  = Query(None),
    date_to:     Optional[str]  = Query(None),
    limit:       int            = Query(config.SEARCH_TOP_K, ge=1, le=50),
    offset:      int            = Query(0, ge=0),
    conn=Depends(get_db),
):
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: _search(q, conn, source_type=source_type, platform=platform,
                        category=category, date_from=date_from, date_to=date_to,
                        limit=limit, offset=offset)
    )
    return {"results": results, "count": len(results)}


@app.get("/status")
def status_endpoint(conn=Depends(get_db)):
    from qdrant_client import QdrantClient
    qdrant = QdrantClient(url=config.QDRANT_URL)
    info = qdrant.get_collection(config.QDRANT_COLLECTION)
    total_chunks  = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    total_sources = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    last_row = conn.execute(
        "SELECT ingested_at FROM sources ORDER BY ingested_at DESC LIMIT 1"
    ).fetchone()
    errors = conn.execute(
        "SELECT COUNT(*) FROM ingestion_errors"
    ).fetchone()[0]
    pending = conn.execute(
        "SELECT COUNT(*) FROM quarantine WHERE status='pending'"
    ).fetchone()[0]
    return {
        "total_chunks":        total_chunks,
        "total_sources":       total_sources,
        "last_ingested":       last_row["ingested_at"] if last_row else None,
        "ingestion_errors":    errors,
        "quarantine_pending":  pending,
        "qdrant_vector_count": info.points_count,
        "qdrant_vector_on_disk": True,
    }


@app.get("/quarantine")
def quarantine_list(status: Optional[str] = Query(None), conn=Depends(get_db)):
    if status:
        rows = conn.execute(
            "SELECT * FROM quarantine WHERE status=? ORDER BY detected_at DESC", [status]
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM quarantine ORDER BY detected_at DESC"
        ).fetchall()
    return {"items": [dict(r) for r in rows]}


@app.post("/quarantine/{quarantine_id}/approve")
def quarantine_approve(quarantine_id: int, background_tasks: BackgroundTasks,
                       conn=Depends(get_db)):
    row = conn.execute("SELECT * FROM quarantine WHERE id=?", [quarantine_id]).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    if row["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"Status is '{row['status']}', not pending")
    conn.execute("UPDATE quarantine SET status='approved' WHERE id=?", [quarantine_id])
    conn.commit()
    return {"status": "approved", "path": row["path"]}


@app.post("/quarantine/{quarantine_id}/reject")
def quarantine_reject(quarantine_id: int, conn=Depends(get_db)):
    row = conn.execute("SELECT * FROM quarantine WHERE id=?", [quarantine_id]).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    conn.execute("UPDATE quarantine SET status='rejected' WHERE id=?", [quarantine_id])
    conn.commit()
    return {"status": "rejected"}
