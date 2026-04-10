# Second Brain — Design Spec
**Date:** 2026-04-09
**Status:** Approved (v5 — post Opus + Gemini review)
**Scope:** v1 — Search mode only. Chat/LLM mode deferred to v2.

---

## 1. Goal

Build a personal semantic search engine ("second brain") over:
- Knowledge folders on the local machine (~50 GB: PDFs, DOCX, notebooks, code, notes)
- Conversation history from Claude, ChatGPT, and Gemini (bulk import + ongoing export watcher)

The system runs entirely on a single machine. No cloud services required. Target platform: macOS Apple Silicon.

> **Ingestion time note:** At ~50 GB, extracted text is typically 1–5 GB (most PDFs, DOCX, code contain far less text than their file size). At ~3,000 tokens/sec on Apple Silicon via Ollama, a 5 GB text corpus (~2B tokens) takes roughly 6–18 hours of continuous embedding. Plan accordingly — run initial ingest overnight.

---

## 2. Architecture

### Components

| Component | Technology | Port / Path |
|---|---|---|
| Search UI | React (built static, served by FastAPI) | localhost:8000/app |
| API backend | FastAPI (single worker — see §3 reranker) | localhost:8000 |
| Vector store | Qdrant ≥ 1.17.1 (installed binary) | localhost:6333 |
| Metadata store | SQLite | `~/qdrant/brain.db` |
| Embedding model | Ollama: `qwen3-embedding:4b` | localhost:11434 |
| Sparse vectors | fastembed `Qdrant/bm25` (in-process, CPU) | n/a |
| Reranker | FlagEmbedding: `BAAI/bge-reranker-v2-m3` (in-process, MPS) | n/a |
| Ingestion pipeline | Python scripts | run manually or via watcher |
| Export watcher | Python daemon | background process |

> **Single-worker requirement:** The FastAPI backend must run with `--workers 1` (see §3 Reranker thread safety).

> **Frontend:** React is built via `npm run build` and served as static files through FastAPI's `StaticFiles`. No client-side routing — all navigation is query-parameter based to avoid SPA 404s.

> **Reranker and sparse vectors:** Both run in-process via Python libraries (FlagEmbedding and fastembed), not through Ollama.

### Directory Layout

```
~/qdrant/
├── qdrant                  # Qdrant binary (v1.17.1)
├── storage/                # Qdrant data
├── brain.db                # SQLite metadata
├── config.py               # pinned model names, paths, constants
├── requirements.txt        # see §2 Dependencies
├── init_schema.py          # idempotent: create Qdrant collection + SQLite schema
├── start.sh                # bring up all processes in correct order
├── stop.sh                 # kill all managed processes via PID files
├── ingestion/
│   ├── ingest_docs.py
│   ├── ingest_convos.py
│   ├── watcher.py
│   ├── parsers/
│   │   ├── base.py
│   │   ├── claude.py
│   │   ├── chatgpt.py
│   │   └── gemini.py
│   └── chunker.py
├── backend/
│   ├── main.py             # FastAPI app + static file serving
│   ├── search.py           # hybrid search + rerank logic
│   └── db.py               # SQLite helpers
├── frontend/               # React app (Vite + TypeScript)
│   └── src/
│       ├── App.tsx
│       ├── SearchBar.tsx
│       ├── FilterPanel.tsx
│       └── ResultsList.tsx
├── tests/
│   ├── fixtures/           # anonymized export samples per platform
│   ├── test_parsers.py
│   └── test_chunker.py
└── docs/superpowers/specs/
    └── 2026-04-09-second-brain-design.md
```

### Dependencies (`requirements.txt`)

```
qdrant-client>=1.10.0       # server-side RRF requires Qdrant client ≥ 1.10
fastembed>=0.3.0
FlagEmbedding>=1.2.0
fastapi
uvicorn
watchdog
pypdf
python-docx
nbformat                    # .ipynb cell parsing
transformers                # Qwen-native tokenizer
sentencepiece               # tokenizer dependency
torch                       # MPS backend for reranker
```

### Startup Orchestration (`start.sh`)

Processes must start in this order:
1. Qdrant binary — wait for port 6333
2. Ollama — wait for port 11434; then warm up: `POST /api/pull {"name": "qwen3-embedding:4b"}`
3. `python init_schema.py` — creates collection + SQLite schema if missing; asserts `meta` matches config if already exists; see §5 for crash-recovery logic
4. FastAPI: `uvicorn backend.main:app --workers 1 --port 8000`
5. Watcher daemon: `python ingestion/watcher.py` (runs startup scan before entering event loop — see §7)

`start.sh` stores PIDs to `~/qdrant/pids/`. `stop.sh` kills by PID file.

React static files are built once (`npm run build` in `frontend/`) before first run. Rebuild only when frontend source changes.

---

## 3. Embedding & Vector Strategy

### Embedding Model
- **Model:** `qwen3-embedding:4b` via Ollama
- **Full output dimension:** 2560
- **Stored/queried dimension:** 512 (Matryoshka truncation + L2-normalize)

```python
import numpy as np

def embed_and_truncate(texts: list[str], instruction: str = "") -> list[list[float]]:
    # /api/embed accepts arrays — use this, NOT /api/embeddings (single-string legacy)
    payload = {"model": EMBEDDING_MODEL, "input": [instruction + t for t in texts]}
    response = ollama_post("/api/embed", payload)
    vecs = response["embeddings"]           # list of 2560-dim float lists
    result = []
    for v in vecs:
        arr = np.array(v[:EMBEDDING_DIM_STORED], dtype=np.float32)
        arr /= (np.linalg.norm(arr) + 1e-9)  # L2-normalize after truncation
        result.append(arr.tolist())
    return result
```

### Asymmetric Query Instructions (Qwen3)

```python
QUERY_INSTRUCTION = "Instruct: Given a question, retrieve relevant passages that answer it\nQuery: "
DOC_INSTRUCTION   = ""   # no prefix for document chunks
```

Usage: `embed_and_truncate([query], instruction=QUERY_INSTRUCTION)` for search; `embed_and_truncate(chunk_texts)` for ingestion.

### Tokenizer

Use the **Qwen-native tokenizer** — not `tiktoken` (OpenAI BPE ≠ Qwen vocabulary):

```python
from transformers import AutoTokenizer
_tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-Embedding-4B",   # size-suffixed ID required; "Qwen/Qwen3-Embedding" does not exist
    trust_remote_code=True,
)

def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text, add_special_tokens=False))

def truncate_tokens(text: str, max_tokens: int) -> str:
    ids = _tokenizer.encode(text, add_special_tokens=False)
    return _tokenizer.decode(ids[:max_tokens])
```

### Qdrant Collection: `brain`

```python
from qdrant_client.models import (
    VectorParams, SparseVectorParams, Distance, HnswConfigDiff,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType,
    Modifier, PayloadSchemaType,
)

client.create_collection(
    collection_name="brain",
    vectors_config={
        "dense": VectorParams(
            size=EMBEDDING_DIM_STORED,    # 512
            distance=Distance.COSINE,
            on_disk=True,
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(modifier=Modifier.IDF)
    },
    quantization_config=ScalarQuantization(        # outer wrapper required
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,
            always_ram=True,
        )
    ),
    on_disk_payload=False,   # payloads are <300 bytes/point; keep in RAM for fast filtering
)

# Payload indexes for all filter + cleanup fields
# Wrapped in try/except because create_payload_index raises if index already exists (idempotency)
for field, schema in [
    ("source_type",  PayloadSchemaType.KEYWORD),
    ("platform",     PayloadSchemaType.KEYWORD),
    ("category",     PayloadSchemaType.KEYWORD),
    ("date",         PayloadSchemaType.DATETIME),   # values must be RFC 3339 (see §9)
    ("source_id",    PayloadSchemaType.INTEGER),    # needed for --sync orphan cleanup
    ("content_hash", PayloadSchemaType.KEYWORD),    # needed for dedup filter queries
]:
    try:
        client.create_payload_index("brain", field_name=field, field_schema=schema)
    except Exception:
        pass  # index already exists — safe to ignore
```

### Hybrid Search: Dense + Sparse (BM25)

fastembed returns a `SparseEmbedding` object; convert explicitly before passing to Qdrant:

```python
from qdrant_client import models
from fastembed import SparseTextEmbedding

bm25_model = SparseTextEmbedding(model_name=BM25_MODEL)

def sparse_embed(text: str) -> models.SparseVector:
    embeddings = list(bm25_model.embed([text]))
    se = embeddings[0]
    return models.SparseVector(
        indices=se.indices.tolist(),
        values=se.values.tolist(),
    )
```

Qdrant server-side RRF (single round trip, dedup handled by server):

```python
from qdrant_client.models import Prefetch, FusionQuery, Fusion

results = client.query_points(
    collection_name="brain",
    prefetch=[
        # Filters MUST be inside each Prefetch — not at the top level.
        # Top-level filter only post-filters the already-fused pool of ~100 candidates,
        # which will return 0 results for narrow categories not in the global top-50.
        Prefetch(query=dense_vec,  using="dense",  limit=RERANK_CANDIDATES, filter=active_filter),
        Prefetch(query=sparse_vec, using="sparse", limit=RERANK_CANDIDATES, filter=active_filter),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=RERANK_CANDIDATES,
    with_payload=True,
)
# active_filter is a models.Filter built from the user's source_type/platform/category/date params
# If no filters are active, pass filter=None in each Prefetch (not omitting the param)
```

### Reranker

- Model: `BAAI/bge-reranker-v2-m3` (XLM-RoBERTa-based cross-encoder)
- **Max sequence length: 512 tokens total** (query + passage). Child chunks are ~900 tokens — too long for the reranker's context window.
- **Solution:** Truncate `chunk_text` to `RERANK_MAX_CHUNK_TOKENS = 400` tokens (using Qwen tokenizer) before scoring. This leaves ~100 tokens of headroom for the query and special tokens.
- Device: `mps` (Apple Silicon); precision: `use_fp16=True`
- **Thread safety:** `FlagReranker` is not thread-safe. The backend runs `--workers 1`. An `asyncio.Lock` guards every `compute_score` call to prevent concurrent access:

```python
import asyncio
from FlagEmbedding import FlagReranker

reranker = FlagReranker(RERANKER_MODEL, use_fp16=RERANKER_FP16, device=RERANKER_DEVICE)
_reranker_lock = asyncio.Lock()

async def rerank(query: str, candidates: list[str]) -> list[float]:
    truncated = [truncate_tokens(c, RERANK_MAX_CHUNK_TOKENS) for c in candidates]
    pairs = [[query, c] for c in truncated]
    async with _reranker_lock:
        scores = await asyncio.get_event_loop().run_in_executor(
            None, lambda: reranker.compute_score(pairs, normalize=True, batch_size=RERANK_BATCH_SIZE)
        )
    return scores

# Warmup in backend startup (hides JIT/model-load latency from first real query):
asyncio.run(rerank("warmup", ["warmup"]))
```

---

## 4. Chunking Strategy

### Documents
- **Child chunks:** target 900 tokens (range 768–1024), 15% overlap (~135 token overlap), measured with Qwen tokenizer
- **Parent storage:** parent text (2.5× child window) stored inline as `parent_text` in SQLite `chunks` — no separate parent table
- Child chunks embedded + stored in Qdrant; `parent_text` fetched for display after search
- Supported formats: `.pdf`, `.docx`, `.txt`, `.md`, `.py`, `.ipynb`
- **`.ipynb` parsing:** use `nbformat` to parse cells; strip all cell outputs (only keep source); chunk markdown and code cells separately by cell boundary before applying token-based chunking within each cell
- **Empty-text handling:** if a file produces zero extractable text (e.g., image-only PDF), log to `ingestion_errors` with `reason="no_extractable_text"` and skip; do not crash
- Category inferred from directory path using `KNOWLEDGE_DIRS` config (each entry has an associated `category` label — see §13)

> **Storage note:** Inline `parent_text` at 2.5× child size creates ~2.5–3× text duplication in SQLite. For a 5 GB extracted-text corpus, expect `brain.db` ~15–20 GB. Accepted v1 tradeoff; v2 can introduce a `parents` table.

### Conversations
- **One turn** = one (user message + assistant response) pair
- Chunked in groups of 1–3 turns per chunk
- If a single assistant message exceeds 1024 tokens (Qwen tokenizer), sub-chunk at paragraph boundaries (`\n\n`) before grouping
- Metadata per chunk: speaker role, platform, conversation title, date

---

## 5. Data Integrity

### Content Hash (Deduplication)
- SHA-256 of whitespace-normalized chunk text (**not lowercased** — case matters for code)
- Normalization: strip leading/trailing whitespace; collapse internal runs of whitespace to single space
- Stored in SQLite `chunks.content_hash`
- Skip re-embedding if `(source_id, chunk_index)` already exists with the **same hash**
- If hash differs → re-embed and update both SQLite row and Qdrant point

### Upsert Key & Qdrant Point ID

```python
import uuid

NAMESPACE_BRAIN = uuid.UUID("7a3f2e1d-4c6b-5a9e-8d7c-1b2a3f4e5d6c")  # fixed — never change

def make_point_id(source_id: int, chunk_index: int) -> str:
    return str(uuid.uuid5(NAMESPACE_BRAIN, f"{source_id}:{chunk_index}"))
```

This namespace is pinned in `config.py`. Changing it orphans all existing Qdrant points.

### SQLite Connection Configuration

Every SQLite connection (in both the ingestion scripts and the FastAPI backend) must execute these PRAGMAs immediately after opening:

```python
conn.execute("PRAGMA foreign_keys = ON")   # enforce ON DELETE CASCADE etc.
conn.execute("PRAGMA journal_mode = WAL")  # allow concurrent reads during writes
conn.execute("PRAGMA busy_timeout = 5000") # wait up to 5s on lock instead of erroring
```

**Why WAL matters:** The two-phase write holds an open SQLite write transaction while waiting for the Qdrant HTTP call to complete. In default rollback-journal mode this write lock blocks all concurrent reads — meaning any search during the 6–18 hour initial ingest will throw `OperationalError: database is locked`. WAL mode allows readers to proceed concurrently with a single writer.

### Two-Phase Write (SQLite + Qdrant)

SQLite transaction wraps both writes:

```python
with sqlite_conn:                          # BEGIN / COMMIT / ROLLBACK
    sqlite_chunk_id = insert_or_update_chunk(...)
    qdrant_client.upsert(
        collection_name="brain",
        points=[PointStruct(id=point_id, vector=..., payload={..., "sqlite_chunk_id": sqlite_chunk_id})]
    )
    # SQLite COMMIT on context-manager exit
    # Qdrant upsert failure → exception → SQLite ROLLBACK
```

**Failure modes:**
- Qdrant succeeds + SQLite COMMIT fails (disk full, lock): Qdrant point exists without SQLite row. A startup reconciliation scan in `init_schema.py` (scroll Qdrant points, check each `sqlite_chunk_id` exists in SQLite, delete orphans) cleans these up. Also triggered manually via `ingest_docs.py --sync`.
- Process killed mid-upsert: same orphan scenario; same cleanup path.

### Re-ingestion Reconciliation
On re-ingest of an edited file:
1. Re-chunk → `new_total` chunks
2. Delete chunks with `chunk_index >= new_total` from Qdrant + SQLite
3. For each remaining chunk: if hash differs → re-embed + upsert; if same → skip
4. Chunk shifting (edits near top of file shift all subsequent chunk indices) is handled automatically — shifted chunks have new hashes and are re-embedded

### Conversation Source Identity
Conversation `sources` rows keyed by platform-native conversation UUID, not file position:

| Platform | Stable ID field |
|---|---|
| Claude | `conversations[i].uuid` |
| ChatGPT | `conversations[i].conversation_id` |
| Gemini | `conversations[i].id` |

Logical `source_path` format: `{platform}::{conversation_id}` (e.g., `claude::a1b2c3d4-...`). Stable across re-downloads of same-named export files.

### Delete Logic
- When a source file is removed: delete all Qdrant points and SQLite rows for that `source_id`
- Triggered by `ingest_docs.py --sync`
- **Safety gate:** `--sync` refuses to delete more than `SYNC_MAX_DELETE_PCT` (10%) of total sources in one run without `--force`, preventing mass deletion from a temporarily unmounted drive

### Embedding Model Versioning & Startup Drift Check

`init_schema.py` runs on every startup and:
1. If `meta` table has no `embedding_model` key (first run or crash mid-init): write keys from current config
2. If `meta` keys exist: assert they match current config:

```python
stored_model = meta.get("embedding_model")
stored_dim   = meta.get("embedding_dim")

if stored_model is None:
    # First run or crashed before meta was written — populate now
    db.execute("INSERT OR REPLACE INTO meta VALUES ('embedding_model', ?)", [EMBEDDING_MODEL])
    db.execute("INSERT OR REPLACE INTO meta VALUES ('embedding_dim', ?)", [str(EMBEDDING_DIM_STORED)])
    db.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '1')")
else:
    assert stored_model == EMBEDDING_MODEL, \
        f"Embedding model mismatch: meta={stored_model}, config={EMBEDDING_MODEL}. Run migration."
    assert int(stored_dim) == EMBEDDING_DIM_STORED, \
        f"Embedding dim mismatch: meta={stored_dim}, config={EMBEDDING_DIM_STORED}. Run migration."
```

### Re-ingestion at Scale
- `ingest_docs.py` writes a checkpoint to `ingestion_progress` after each successful batch
- On restart: skip chunks where hash + model version match stored values
- Progress visible via `/status` endpoint

---

## 6. Ingestion Pipeline

### `ingest_docs.py`

```
Usage:
  python ingestion/ingest_docs.py           # ingest all KNOWLEDGE_DIRS
  python ingestion/ingest_docs.py --sync    # ingest + reconcile removed files + Qdrant orphans
  python ingestion/ingest_docs.py --dry-run # show what would change, no writes
```

Steps:
1. Walk `KNOWLEDGE_DIRS`, apply allowlist/denylist, skip files > `MAX_FILE_SIZE_BYTES`
2. Extract text per file type; skip (log to `ingestion_errors`) if zero text
3. Chunk with child/parent strategy
4. Compute SHA-256 per child chunk (whitespace-normalized, not lowercased)
5. Embed child chunks via Ollama `/api/embed` in batches of `EMBED_BATCH_SIZE=32`
6. Produce BM25 sparse vectors via fastembed in same batches
7. Two-phase write: SQLite transaction → Qdrant upsert → commit
8. Write `ingestion_progress` checkpoint after each batch
9. If `--sync`: compare SQLite `sources` vs disk; delete orphaned sources (10% safety gate); scroll Qdrant for orphan points (no SQLite row); delete them

**Ollama retry:** 3 attempts, exponential backoff (1s, 2s, 4s). On permanent failure: log to `ingestion_errors`, continue.

### `ingest_convos.py`

```
Usage:
  python ingestion/ingest_convos.py path/to/export.json [path2 ...]
```

Steps:
1. Detect platform (see Platform Detection)
2. Parse → `List[Conversation]`
3. For each conversation: look up or create `sources` row keyed by `{platform}::{conversation_id}`
4. Chunk by message-group boundaries
5. Two-phase write + batch/retry (same as `ingest_docs.py`)

**Checkpoint:** `ingestion_progress` is keyed by `file_path` (the export file path, not a `source_id` — because the file itself is not a source row). `last_convo_index` = index of last fully ingested conversation. On resume, skip conversations ≤ `last_convo_index`. `last_chunk_index` tracks within-conversation chunk progress. Both updated atomically after each conversation. After full ingestion, set `quarantine.ingested_count` = total conversations extracted.

### Platform Detection

| Platform | Detection heuristic |
|---|---|
| Claude | Top-level key `"conversations"` (object, not array); items contain `"uuid"` and `"chat_messages"` |
| ChatGPT | Top-level is a JSON **array**; items contain `"mapping"` and `"conversation_id"` |
| Gemini | Top-level key `"conversations"` (object); items contain `"id"` and `"reply"` |

No Claude/ChatGPT collision possible (object vs array). Unknown format → reject, log to `quarantine` with `reason="unknown_format"`.

### Parser Interface (`ingestion/parsers/base.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class Message:
    role: str
    content: str
    timestamp: Optional[str]   # ISO 8601 or None

@dataclass
class Conversation:
    platform: str              # "claude" | "chatgpt" | "gemini"
    conversation_id: str       # platform-native stable UUID/ID
    title: str
    date: str                  # ISO 8601 date of first message
    messages: List[Message]

class Parser(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> List[Conversation]:
        ...
```

### Snapshot Tests
- One fixture file per platform in `tests/fixtures/` (small, anonymized)
- Each test asserts `parser.parse(fixture)` matches stored JSON snapshot
- When an export format changes, exactly one test fails

---

## 7. Export Watcher Daemon (`ingestion/watcher.py`)

Watches `~/Downloads` using `watchdog`.

> **Knowledge directory sync:** `KNOWLEDGE_DIRS` not watched live. Run `ingest_docs.py --sync` manually or via launchd/cron. Deliberate v1 scope decision.

### Startup Scan (Missed Files)

Before entering the `watchdog` event loop, the daemon scans `WATCHER_WATCH_DIR` for existing `.json` files and cross-checks against `quarantine.path`. Any file not already in `quarantine` is enqueued as if freshly detected. This handles files downloaded while the daemon was offline (reboot, crash):

```python
def startup_scan():
    existing = list(Path(WATCHER_WATCH_DIR).glob("*.json"))
    known_paths = set(db.query("SELECT path FROM quarantine"))
    for f in existing:
        if str(f) not in known_paths and f.stat().st_size <= MAX_FILE_SIZE_BYTES:
            enqueue_for_review(f)
```

### File-Complete Check (Race Condition)

Browsers write downloads as incomplete temporaries. Poll until stable:

```python
def wait_for_file_complete(path: Path, stable_secs: float = 3.0, timeout_secs: float = 60.0) -> bool:
    import time
    deadline = time.time() + timeout_secs
    last_size, stable_since = -1, None
    while time.time() < deadline:
        size = path.stat().st_size
        if size == last_size:
            if stable_since and (time.time() - stable_since) >= stable_secs:
                return True
        else:
            stable_since = time.time()
        last_size = size
        time.sleep(0.5)
    return False
```

Only enqueue a file after `wait_for_file_complete` returns `True`.

### Allowlist / Denylist / Size Cap
- **Watcher allowlist:** `.json` only
- **Doc allowlist:** `.pdf`, `.docx`, `.txt`, `.md`, `.py`, `.ipynb`
- **Denylist:** `node_modules`, `.git`, `venv`, `.venv`, `__pycache__`, `env`, `dist`, `build`, `.next`, `site-packages`, `tmp`
- **Max file size:** 200 MB

### Quarantine Queue

1. Watcher detects + verifies complete file → insert into `quarantine` with `status='pending'`
2. FastAPI `/quarantine` exposes queue; React UI shows review panel
3. User approves via UI → `POST /quarantine/{id}/approve`
4. **The FastAPI endpoint** (not the watcher) triggers ingestion via `BackgroundTasks`:

```python
@app.post("/quarantine/{quarantine_id}/approve")
async def approve(quarantine_id: int, background_tasks: BackgroundTasks, db=Depends(get_db)):
    row = db.get_quarantine(quarantine_id)
    db.update_quarantine_status(quarantine_id, "approved")
    background_tasks.add_task(run_ingest_convos, row["path"])
    return {"status": "approved", "ingesting": True}
```

5. After successful ingestion → `status='ingested'`, `ingested_at` set, `source_id` FK set
6. On rejection → `status='rejected'`; row kept for audit

---

## 8. SQLite Schema (`brain.db`)

```sql
CREATE TABLE sources (
    id           INTEGER PRIMARY KEY,
    path         TEXT UNIQUE NOT NULL,   -- "{platform}::{conv_id}" or filesystem path
    source_type  TEXT NOT NULL,          -- 'document' | 'conversation'
    platform     TEXT,                   -- 'local' | 'claude' | 'chatgpt' | 'gemini'
    title        TEXT,
    date         TEXT,                   -- RFC 3339: "2026-04-09T00:00:00Z"
    ingested_at  TEXT NOT NULL,
    file_hash    TEXT
);

CREATE TABLE chunks (
    id                      INTEGER PRIMARY KEY,
    source_id               INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    qdrant_point_id         TEXT NOT NULL UNIQUE,
    chunk_index             INTEGER NOT NULL,
    chunk_text              TEXT NOT NULL,
    parent_text             TEXT,
    content_hash            TEXT NOT NULL,    -- SHA-256, whitespace-normalized, not lowercased
    embedding_model_version TEXT NOT NULL,
    UNIQUE(source_id, chunk_index)
);

CREATE TABLE quarantine (
    id             INTEGER PRIMARY KEY,
    path           TEXT NOT NULL,
    detected_at    TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'pending',  -- 'pending'|'approved'|'rejected'|'ingested'
    ingested_at    TEXT,
    ingested_count INTEGER,   -- how many conversations/documents were extracted from this file
    reason         TEXT
);

CREATE TABLE ingestion_errors (
    id          INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL,
    error       TEXT NOT NULL,
    occurred_at TEXT NOT NULL
);

CREATE TABLE ingestion_progress (
    -- Keyed by file_path (not source_id) because export files are not themselves sources;
    -- each conversation within a file becomes its own sources row.
    file_path            TEXT PRIMARY KEY,
    last_chunk_index     INTEGER NOT NULL,
    last_convo_index     INTEGER,    -- conversation exports: last fully ingested conversation index
    updated_at           TEXT NOT NULL
);

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- Populated by init_schema.py:
--   embedding_model  → 'qwen3-embedding:4b'
--   embedding_dim    → '512'
--   schema_version   → '1'

-- Indexes
CREATE INDEX idx_chunks_source_id        ON chunks(source_id);
CREATE INDEX idx_chunks_content_hash     ON chunks(content_hash);
CREATE INDEX idx_chunks_qdrant_point_id  ON chunks(qdrant_point_id);
CREATE INDEX idx_chunks_emb_version      ON chunks(embedding_model_version);
CREATE INDEX idx_sources_path            ON sources(path);
CREATE INDEX idx_quarantine_status       ON quarantine(status);
```

---

## 9. Qdrant Point Payload

Each Qdrant point carries filterable metadata (no raw text — text lives in SQLite):

```json
{
  "source_type":     "document | conversation",
  "platform":        "local | claude | chatgpt | gemini",
  "title":           "...",
  "date":            "2026-04-09T00:00:00Z",
  "category":        "AI-Research | LLM-Training-Framework | ...",
  "source_path":     "claude::a1b2c3... or /Users/.../file.pdf",
  "chunk_index":     3,
  "content_hash":    "sha256hex...",
  "source_id":       42,
  "sqlite_chunk_id": 1234
}
```

> **Date format:** All `date` values must be **RFC 3339** (`2026-04-09T00:00:00Z`) — not date-only strings. Qdrant's DATETIME payload index rejects bare date strings. When only a date is known (e.g., file mtime), append `T00:00:00Z`.

---

## 10. Search Pipeline

```
User query string
    │
    ▼
Apply QUERY_INSTRUCTION prefix
    │
    ▼
Ollama /api/embed (array input) → 2560-dim
    │
    ▼
Slice [:512] → L2-normalize → dim=512 dense vector
    │                                         │
    ▼                                         ▼
fastembed BM25 sparse vector             (same query)
→ models.SparseVector(indices, values)
    │                                         │
    └──── Qdrant server-side RRF fusion ──────┘
          (Prefetch dense + sparse, FusionQuery RRF)
          active_filter pushed into BOTH Prefetch objects
          top-50 candidates, dedup server-side
                      │
                      ▼
    Fetch chunk_text from SQLite (by sqlite_chunk_id)
                      │
                      ▼
    Truncate each chunk_text to RERANK_MAX_CHUNK_TOKENS (400)
    using Qwen tokenizer (bge-reranker-v2-m3 max = 512 tokens total)
                      │
                      ▼
    bge-reranker-v2-m3 cross-encoder (FlagEmbedding, MPS, fp16)
    score each (query_text, truncated_chunk_text) pair
    batched at RERANK_BATCH_SIZE=16, guarded by asyncio.Lock
                      │
                      ▼
              top-10 by reranker score
                      │
                      ▼
    Fetch parent_text from SQLite for display
                      │
                      ▼
              Return to API → React UI
```

---

## 11. FastAPI Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/search` | Hybrid search with filters |
| `GET` | `/status` | Ingestion stats + Qdrant collection health |
| `GET` | `/quarantine` | List quarantine queue (filterable by status) |
| `POST` | `/quarantine/{id}/approve` | Approve → triggers `ingest_convos.py` via BackgroundTasks |
| `POST` | `/quarantine/{id}/reject` | Reject a file |

### `/search` Query Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `q` | string | required | Search query |
| `source_type` | string | all | `document` or `conversation` |
| `platform` | string | all | `local`, `claude`, `chatgpt`, `gemini` |
| `category` | string | all | Filter by category label |
| `date_from` | string | none | RFC 3339 datetime |
| `date_to` | string | none | RFC 3339 datetime |
| `limit` | int | 10 | Max results (max 50) |
| `offset` | int | 0 | Pagination offset |

### `/status` Response Includes
- Total chunks, total sources, last ingested timestamp
- Active `ingestion_errors` count
- Qdrant collection vector count and on-disk storage size
- Quarantine pending count

### Search Result Shape

```json
{
  "score":       0.92,
  "title":       "Mechanistic Interpretability Lecture 5",
  "platform":    "local",
  "category":    "AI-Research",
  "date":        "2025-11-01T00:00:00Z",
  "source_path": "/Users/.../5- Mechanistic Interpretability.docx",
  "chunk_text":  "...child chunk snippet...",
  "parent_text": "...broader context window..."
}
```

---

## 12. React UI

### Two-panel layout
- **Left:** search bar + filter panel (source_type, platform, category, date range)
- **Right:** ranked results list

### Result card
- Title, platform badge (color-coded), date, category tag
- Child chunk snippet
- "Show context" toggle → expands `parent_text`
- Relevance score indicator

### States
- **Loading:** spinner on results panel (expected latency 3–8 s including reranker on MPS)
- **Error:** banner when backend is unreachable or returns 5xx
- **Empty:** "No results found. Try broader filters or a different query."
- **Pagination:** load-more button using `offset` param — no infinite scroll

> **No client-side routing.** All state (filters, query, page) is encoded in URL query parameters so FastAPI's static file handler serves `index.html` for every path without a catch-all SPA fallback.

---

## 13. Config (`config.py`)

```python
import uuid

# Models
EMBEDDING_MODEL           = "qwen3-embedding:4b"
EMBEDDING_TOKENIZER_ID    = "Qwen/Qwen3-Embedding-4B"  # HF model ID for tokenizer
EMBEDDING_DIM_FULL        = 2560
EMBEDDING_DIM_STORED      = 512
RERANKER_MODEL            = "BAAI/bge-reranker-v2-m3"
RERANKER_DEVICE           = "mps"
RERANKER_FP16             = True
RERANK_MAX_CHUNK_TOKENS   = 400   # bge-reranker-v2-m3 max = 512 total; 400 leaves room for query
RERANK_BATCH_SIZE         = 16
BM25_MODEL                = "Qdrant/bm25"
OLLAMA_URL                = "http://localhost:11434"
QDRANT_URL                = "http://localhost:6333"
QDRANT_COLLECTION         = "brain"
SQLITE_PATH               = "/Users/hariramanpokhrel/qdrant/brain.db"

# Deterministic Qdrant point ID namespace — NEVER regenerate this UUID
NAMESPACE_BRAIN           = uuid.UUID("7a3f2e1d-4c6b-5a9e-8d7c-1b2a3f4e5d6c")

# Embedding instructions
QUERY_INSTRUCTION         = "Instruct: Given a question, retrieve relevant passages that answer it\nQuery: "
DOC_INSTRUCTION           = ""

# Chunking
CHUNK_SIZE_TOKENS         = 900
CHUNK_OVERLAP_PCT         = 0.15
PARENT_WINDOW_MULT        = 2.5

# Search & ingestion
EMBED_BATCH_SIZE          = 32
RERANK_CANDIDATES         = 50
SEARCH_TOP_K              = 10
OLLAMA_RETRY_ATTEMPTS     = 3
OLLAMA_RETRY_BACKOFF      = [1, 2, 4]

# Watcher / safety
MAX_FILE_SIZE_BYTES       = 200 * 1024 * 1024
SYNC_MAX_DELETE_PCT       = 0.10
WATCHER_WATCH_DIR         = "/Users/hariramanpokhrel/Downloads"
WATCHER_ALLOWLIST         = {".json"}
DOC_ALLOWLIST             = {".pdf", ".docx", ".txt", ".md", ".py", ".ipynb"}
WATCHER_DENYLIST          = {
    "node_modules", ".git", "venv", ".venv", "__pycache__",
    "env", "dist", "build", ".next", "site-packages", "tmp",
}

# Knowledge directories with category labels
KNOWLEDGE_DIRS = [
    ("/Users/hariramanpokhrel/Documents/BuildAnLLM",          "LLM-Training-Framework"),
    ("/Users/hariramanpokhrel/Documents/MIT Deep Learning",   "MIT-Deep-Learning-Course"),
    ("/Users/hariramanpokhrel/Documents/AlgoVerse Research",  "AI-Research"),
    ("/Users/hariramanpokhrel/Documents/nanochat",            "LLM-Chat-Framework"),
    ("/Users/hariramanpokhrel/Documents/JupyterNotebook",     "Notebooks"),
    ("/Users/hariramanpokhrel/Desktop/Augmented-Learning",    "AI-Education"),
    ("/Users/hariramanpokhrel/Desktop/autoresearchclaw",      "Research-Automation"),
    ("/Users/hariramanpokhrel/Desktop/PaperVerifier",         "Paper-Verification"),
    ("/Users/hariramanpokhrel/Desktop/privatellmgateway",     "LLM-Gateway"),
    ("/Users/hariramanpokhrel/Desktop/Wonderful-Learning",    "Deep-Learning-Education"),
    ("/Users/hariramanpokhrel/Desktop/claw-code",             "AI-Research-Framework"),
    ("/Users/hariramanpokhrel/Desktop/playground",            "AI-Research"),
]
```

---

## 14. Out of Scope (v1)

- Chat / RAG mode with LLM response generation (v2)
- Multi-machine / server deployment
- User authentication
- Image, audio, or video ingestion
- OCR for image-only PDFs
- Real-time browser extension capture
- Automatic conversation scraping (export watcher only)
- Live sync of `KNOWLEDGE_DIRS` (batch `--sync` only in v1)
- Multi-model embedding migration tooling (v2)
