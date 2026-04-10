# Second Brain Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local semantic search engine over ~50 GB of personal documents and AI conversation exports using Qdrant, FastAPI, and React.

**Architecture:** Ingestion pipeline chunks documents and conversation exports, embeds them with Qwen3-Embedding-4B via Ollama at dim=512 (Matryoshka + L2-normalize), and upserts to Qdrant with BM25 sparse vectors. FastAPI serves hybrid search (server-side RRF fusion) with bge-reranker-v2-m3 cross-encoder reranking. React static UI served by FastAPI.

**Tech Stack:** Python 3.11+, Qdrant 1.17.1, SQLite (WAL mode), FastAPI, Ollama, fastembed, FlagEmbedding, transformers (Qwen tokenizer), React + Vite + TypeScript.

---

## Chunk 1: Foundation — Config, Schema, Init

### File Map

| File | Responsibility |
|---|---|
| `config.py` | All constants — model names, paths, thresholds |
| `requirements.txt` | Pinned dependencies |
| `backend/db.py` | SQLite connection factory + PRAGMA setup + all query helpers |
| `init_schema.py` | Idempotent: create Qdrant collection + SQLite tables + payload indexes + meta assertion |
| `start.sh` | Ordered startup: Qdrant → Ollama warmup → init_schema → uvicorn → watcher |
| `stop.sh` | Kill all PIDs |

---

### Task 1: `config.py`

**Files:**
- Create: `~/qdrant/config.py`

- [ ] **Step 1: Create config.py**

```python
# ~/qdrant/config.py
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ── Models ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL        = "qwen3-embedding:4b"
EMBEDDING_TOKENIZER_ID = "Qwen/Qwen3-Embedding-4B"   # HF repo — size-suffix required
EMBEDDING_DIM_FULL     = 2560
EMBEDDING_DIM_STORED   = 512                          # Matryoshka truncation dim
RERANKER_MODEL         = "BAAI/bge-reranker-v2-m3"
RERANKER_DEVICE        = "mps"
RERANKER_FP16          = True
RERANK_MAX_CHUNK_TOKENS = 400   # bge-reranker max=512 total; 400 leaves room for query
RERANK_BATCH_SIZE      = 16
BM25_MODEL             = "Qdrant/bm25"

# ── Services ──────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434"
QDRANT_URL   = "http://localhost:6333"
QDRANT_COLLECTION = "brain"
SQLITE_PATH  = str(BASE_DIR / "brain.db")

# ── Qdrant point ID namespace — NEVER change this UUID ───────────────────────
NAMESPACE_BRAIN = uuid.UUID("7a3f2e1d-4c6b-5a9e-8d7c-1b2a3f4e5d6c")

# ── Embedding instructions (asymmetric) ───────────────────────────────────────
QUERY_INSTRUCTION = (
    "Instruct: Given a question, retrieve relevant passages that answer it\nQuery: "
)
DOC_INSTRUCTION = ""

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE_TOKENS  = 900
CHUNK_OVERLAP_PCT  = 0.15
PARENT_WINDOW_MULT = 2.5

# ── Search & ingestion ────────────────────────────────────────────────────────
EMBED_BATCH_SIZE      = 32
RERANK_CANDIDATES     = 50
SEARCH_TOP_K          = 10
OLLAMA_RETRY_ATTEMPTS = 3
OLLAMA_RETRY_BACKOFF  = [1, 2, 4]

# ── Watcher / safety ─────────────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES  = 200 * 1024 * 1024   # 200 MB
SYNC_MAX_DELETE_PCT  = 0.10                # refuse to delete >10% of sources without --force
WATCHER_WATCH_DIR    = str(Path.home() / "Downloads")
WATCHER_ALLOWLIST    = {".json"}
DOC_ALLOWLIST        = {".pdf", ".docx", ".txt", ".md", ".py", ".ipynb"}
WATCHER_DENYLIST     = {
    "node_modules", ".git", "venv", ".venv", "__pycache__",
    "env", "dist", "build", ".next", "site-packages", "tmp",
}

# ── Knowledge directories (path, category label) ─────────────────────────────
KNOWLEDGE_DIRS = [
    (str(Path.home() / "Documents/BuildAnLLM"),         "LLM-Training-Framework"),
    (str(Path.home() / "Documents/MIT Deep Learning"),  "MIT-Deep-Learning-Course"),
    (str(Path.home() / "Documents/AlgoVerse Research"), "AI-Research"),
    (str(Path.home() / "Documents/nanochat"),           "LLM-Chat-Framework"),
    (str(Path.home() / "Documents/JupyterNotebook"),    "Notebooks"),
    (str(Path.home() / "Desktop/Augmented-Learning"),   "AI-Education"),
    (str(Path.home() / "Desktop/autoresearchclaw"),     "Research-Automation"),
    (str(Path.home() / "Desktop/PaperVerifier"),        "Paper-Verification"),
    (str(Path.home() / "Desktop/privatellmgateway"),    "LLM-Gateway"),
    (str(Path.home() / "Desktop/Wonderful-Learning"),   "Deep-Learning-Education"),
    (str(Path.home() / "Desktop/claw-code"),            "AI-Research-Framework"),
    (str(Path.home() / "Desktop/playground"),           "AI-Research"),
]
```

- [ ] **Step 2: Verify syntax**
```bash
cd ~/qdrant && python -c "import config; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**
```bash
cd ~/qdrant && git init && git add config.py && git commit -m "feat: add config.py with all constants"
```

---

### Task 2: `requirements.txt`

**Files:**
- Create: `~/qdrant/requirements.txt`

- [ ] **Step 1: Create requirements.txt**

```
qdrant-client>=1.10.0
fastembed>=0.3.0
FlagEmbedding>=1.2.0
fastapi
uvicorn[standard]
watchdog
pypdf
python-docx
nbformat
transformers>=4.40.0
sentencepiece
torch
numpy
httpx
pytest
pytest-asyncio
```

- [ ] **Step 2: Install**
```bash
cd ~/qdrant && pip install -r requirements.txt
```
Expected: All packages install without error.

- [ ] **Step 3: Commit**
```bash
git add requirements.txt && git commit -m "feat: add requirements.txt"
```

---

### Task 3: `backend/db.py` — SQLite connection factory

**Files:**
- Create: `~/qdrant/backend/__init__.py` (empty)
- Create: `~/qdrant/backend/db.py`

- [ ] **Step 1: Write failing test**

Create `~/qdrant/tests/test_db.py`:

```python
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
    assert jm == "wal", "journal_mode must be WAL"

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
```

- [ ] **Step 2: Run test — expect FAIL**
```bash
cd ~/qdrant && python -m pytest tests/test_db.py -v
```
Expected: `ModuleNotFoundError: No module named 'backend.db'`

- [ ] **Step 3: Create `backend/__init__.py`**
```bash
touch ~/qdrant/backend/__init__.py
```

- [ ] **Step 4: Create `backend/db.py`**

```python
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
```

- [ ] **Step 5: Run test — expect PASS**
```bash
cd ~/qdrant && python -m pytest tests/test_db.py -v
```
Expected: All 3 tests PASS.

- [ ] **Step 6: Commit**
```bash
git add backend/__init__.py backend/db.py tests/test_db.py && \
git commit -m "feat: SQLite connection factory with WAL mode and full schema DDL"
```

---

### Task 4: `init_schema.py`

**Files:**
- Create: `~/qdrant/init_schema.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_db.py`:

```python
def test_meta_assertion_passes_on_match():
    """init_schema should not raise when meta matches config."""
    conn = get_connection()
    init_tables(conn)
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('embedding_model', ?)", [config.EMBEDDING_MODEL])
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('embedding_dim', ?)", [str(config.EMBEDDING_DIM_STORED)])
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '1')")
    conn.commit()
    # Should not raise
    from init_schema import assert_meta_matches
    assert_meta_matches(conn)

def test_meta_assertion_raises_on_mismatch():
    import pytest
    conn = get_connection()
    init_tables(conn)
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('embedding_model', 'old-model')")
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('embedding_dim', '999')")
    conn.commit()
    from init_schema import assert_meta_matches
    with pytest.raises(AssertionError):
        assert_meta_matches(conn)
```

- [ ] **Step 2: Run — expect FAIL**
```bash
cd ~/qdrant && python -m pytest tests/test_db.py::test_meta_assertion_passes_on_match -v
```
Expected: `ModuleNotFoundError: No module named 'init_schema'`

- [ ] **Step 3: Create `init_schema.py`**

```python
#!/usr/bin/env python3
# ~/qdrant/init_schema.py
"""
Idempotent setup: creates Qdrant collection + SQLite schema.
Run on every startup — safe to re-run.
"""
import sys
import config
from backend.db import get_connection, init_tables

def assert_meta_matches(conn) -> None:
    rows = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM meta")}
    stored_model = rows.get("embedding_model")
    stored_dim   = rows.get("embedding_dim")
    if stored_model is None:
        # First run or crashed before meta write — populate now
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('embedding_model', ?)", [config.EMBEDDING_MODEL])
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('embedding_dim', ?)",   [str(config.EMBEDDING_DIM_STORED)])
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '1')")
        conn.commit()
        print(f"[init_schema] meta written: model={config.EMBEDDING_MODEL} dim={config.EMBEDDING_DIM_STORED}")
        return
    assert stored_model == config.EMBEDDING_MODEL, (
        f"Embedding model mismatch: meta='{stored_model}' config='{config.EMBEDDING_MODEL}'. "
        "Run a migration or update config."
    )
    assert int(stored_dim) == config.EMBEDDING_DIM_STORED, (
        f"Embedding dim mismatch: meta={stored_dim} config={config.EMBEDDING_DIM_STORED}. "
        "Run a migration or update config."
    )
    print(f"[init_schema] meta OK: model={stored_model} dim={stored_dim}")


def setup_qdrant() -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        VectorParams, SparseVectorParams, Distance, HnswConfigDiff,
        ScalarQuantization, ScalarQuantizationConfig, ScalarType,
        Modifier, PayloadSchemaType,
    )
    client = QdrantClient(url=config.QDRANT_URL)
    existing = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config={
                "dense": VectorParams(
                    size=config.EMBEDDING_DIM_STORED,
                    distance=Distance.COSINE,
                    on_disk=True,
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(modifier=Modifier.IDF)
            },
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            ),
            on_disk_payload=False,
        )
        print(f"[init_schema] Qdrant collection '{config.QDRANT_COLLECTION}' created.")
    else:
        print(f"[init_schema] Qdrant collection '{config.QDRANT_COLLECTION}' already exists.")

    # Payload indexes — idempotent (ignore if already exists)
    for field, schema in [
        ("source_type",  PayloadSchemaType.KEYWORD),
        ("platform",     PayloadSchemaType.KEYWORD),
        ("category",     PayloadSchemaType.KEYWORD),
        ("date",         PayloadSchemaType.DATETIME),
        ("source_id",    PayloadSchemaType.INTEGER),
        ("content_hash", PayloadSchemaType.KEYWORD),
    ]:
        try:
            client.create_payload_index(config.QDRANT_COLLECTION,
                                        field_name=field, field_schema=schema)
        except Exception:
            pass


def main() -> None:
    print("[init_schema] Starting...")
    conn = get_connection()
    init_tables(conn)
    assert_meta_matches(conn)
    setup_qdrant()
    print("[init_schema] Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run meta tests — expect PASS**
```bash
cd ~/qdrant && python -m pytest tests/test_db.py -v
```
Expected: All 5 tests PASS.

- [ ] **Step 5: Smoke-test against live Qdrant**
```bash
cd ~/qdrant && python init_schema.py
```
Expected output:
```
[init_schema] Starting...
[init_schema] meta written: model=qwen3-embedding:4b dim=512
[init_schema] Qdrant collection 'brain' created.
[init_schema] Done.
```

Run again — should be idempotent:
```bash
python init_schema.py
```
Expected: No errors, `[init_schema] meta OK` and `collection 'brain' already exists`.

- [ ] **Step 6: Commit**
```bash
git add init_schema.py tests/test_db.py && \
git commit -m "feat: idempotent init_schema with meta assertion and Qdrant collection setup"
```

---

### Task 5: `start.sh` / `stop.sh`

**Files:**
- Create: `~/qdrant/start.sh`
- Create: `~/qdrant/stop.sh`

- [ ] **Step 1: Create `start.sh`**

```bash
#!/usr/bin/env bash
# ~/qdrant/start.sh
set -euo pipefail
BASE="$(cd "$(dirname "$0")" && pwd)"
PIDS="$BASE/pids"
mkdir -p "$PIDS"

echo "==> Starting Qdrant..."
"$BASE/qdrant" --config-path "$BASE/config.yaml" &>/tmp/qdrant.log &
echo $! > "$PIDS/qdrant.pid"
until curl -sf http://localhost:6333/healthz &>/dev/null; do sleep 0.5; done
echo "    Qdrant up."

echo "==> Warming up Ollama model..."
curl -sf -X POST http://localhost:11434/api/pull \
  -d "{\"name\": \"$(cd "$BASE" && python -c 'import config; print(config.EMBEDDING_MODEL)')\"}" \
  -H "Content-Type: application/json" | tail -1
echo "    Ollama ready."

echo "==> Running init_schema..."
cd "$BASE" && python init_schema.py

echo "==> Starting FastAPI backend..."
uvicorn backend.main:app --workers 1 --port 8000 --host 127.0.0.1 \
  &>/tmp/brain-backend.log &
echo $! > "$PIDS/backend.pid"
until curl -sf http://localhost:8000/status &>/dev/null; do sleep 0.5; done
echo "    Backend up."

echo "==> Starting watcher daemon..."
python "$BASE/ingestion/watcher.py" &>/tmp/brain-watcher.log &
echo $! > "$PIDS/watcher.pid"

echo ""
echo "Second Brain running:"
echo "  Search UI: http://localhost:8000/app"
echo "  API:       http://localhost:8000"
echo "  Logs:      /tmp/qdrant.log  /tmp/brain-backend.log  /tmp/brain-watcher.log"
```

- [ ] **Step 2: Create `stop.sh`**

```bash
#!/usr/bin/env bash
BASE="$(cd "$(dirname "$0")" && pwd)"
PIDS="$BASE/pids"
for f in "$PIDS"/*.pid; do
  [ -f "$f" ] || continue
  pid=$(cat "$f")
  kill "$pid" 2>/dev/null && echo "Stopped PID $pid ($(basename "$f" .pid))" || true
  rm "$f"
done
echo "All processes stopped."
```

- [ ] **Step 3: Make executable and commit**
```bash
chmod +x ~/qdrant/start.sh ~/qdrant/stop.sh && \
git add start.sh stop.sh && \
git commit -m "feat: start.sh and stop.sh for process orchestration"
```

---

## Chunk 2: Embedding, Tokenizer, Chunker

### File Map

| File | Responsibility |
|---|---|
| `ingestion/embedder.py` | Ollama `/api/embed` wrapper + L2-normalize + retry |
| `ingestion/sparse.py` | fastembed BM25 wrapper → `models.SparseVector` |
| `ingestion/chunker.py` | Qwen tokenizer + child/parent chunking for docs and conversations |

---

### Task 6: `ingestion/embedder.py`

**Files:**
- Create: `~/qdrant/ingestion/__init__.py` (empty)
- Create: `~/qdrant/ingestion/embedder.py`
- Create: `~/qdrant/tests/__init__.py` (empty)
- Test: `~/qdrant/tests/test_embedder.py`

- [ ] **Step 1: Write failing test**

```python
# ~/qdrant/tests/test_embedder.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

def test_embed_returns_correct_dim():
    """Integration test: requires Ollama running with qwen3-embedding:4b."""
    from ingestion.embedder import embed_texts
    vecs = embed_texts(["hello world", "test sentence"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 512   # EMBEDDING_DIM_STORED
    assert len(vecs[1]) == 512

def test_embed_is_unit_normalized():
    from ingestion.embedder import embed_texts
    vecs = embed_texts(["normalize me"])
    arr = np.array(vecs[0])
    norm = np.linalg.norm(arr)
    assert abs(norm - 1.0) < 1e-5, f"vector not unit-normalized, norm={norm}"

def test_embed_with_instruction():
    from ingestion.embedder import embed_texts
    import config
    vecs = embed_texts(["what is attention?"], instruction=config.QUERY_INSTRUCTION)
    assert len(vecs[0]) == 512
```

- [ ] **Step 2: Run — expect FAIL**
```bash
cd ~/qdrant && python -m pytest tests/test_embedder.py -v
```
Expected: `ModuleNotFoundError: No module named 'ingestion.embedder'`

- [ ] **Step 3: Create `ingestion/__init__.py`**
```bash
touch ~/qdrant/ingestion/__init__.py ~/qdrant/tests/__init__.py
```

- [ ] **Step 4: Create `ingestion/embedder.py`**

```python
# ~/qdrant/ingestion/embedder.py
import time
import httpx
import numpy as np
from typing import Optional
import config

def embed_texts(
    texts: list[str],
    instruction: str = config.DOC_INSTRUCTION,
    model: str = config.EMBEDDING_MODEL,
) -> list[list[float]]:
    """Embed texts via Ollama /api/embed, truncate to dim=512, L2-normalize."""
    inputs = [instruction + t for t in texts]
    payload = {"model": model, "input": inputs}

    last_exc = None
    for attempt, delay in enumerate([0] + config.OLLAMA_RETRY_BACKOFF):
        if delay:
            time.sleep(delay)
        try:
            resp = httpx.post(
                f"{config.OLLAMA_URL}/api/embed",
                json=payload,
                timeout=120.0,
            )
            resp.raise_for_status()
            raw_vecs = resp.json()["embeddings"]
            return [_truncate_normalize(v) for v in raw_vecs]
        except Exception as exc:
            last_exc = exc
            if attempt < len(config.OLLAMA_RETRY_BACKOFF):
                continue
    raise RuntimeError(f"Ollama embed failed after retries: {last_exc}") from last_exc


def _truncate_normalize(vec: list[float]) -> list[float]:
    arr = np.array(vec[: config.EMBEDDING_DIM_STORED], dtype=np.float32)
    norm = np.linalg.norm(arr)
    arr = arr / (norm + 1e-9)
    return arr.tolist()
```

- [ ] **Step 5: Run tests — expect PASS** (requires Ollama running)
```bash
cd ~/qdrant && python -m pytest tests/test_embedder.py -v
```
Expected: All 3 PASS.

- [ ] **Step 6: Commit**
```bash
git add ingestion/__init__.py ingestion/embedder.py tests/__init__.py tests/test_embedder.py && \
git commit -m "feat: Ollama embedder with Matryoshka truncation and L2-normalize"
```

---

### Task 7: `ingestion/sparse.py`

**Files:**
- Create: `~/qdrant/ingestion/sparse.py`
- Test: `~/qdrant/tests/test_sparse.py`

- [ ] **Step 1: Write failing test**

```python
# ~/qdrant/tests/test_sparse.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_sparse_returns_qdrant_type():
    from ingestion.sparse import sparse_embed
    from qdrant_client import models
    sv = sparse_embed("hello world attention mechanism")
    assert isinstance(sv, models.SparseVector)
    assert len(sv.indices) > 0
    assert len(sv.indices) == len(sv.values)

def test_sparse_indices_are_ints():
    from ingestion.sparse import sparse_embed
    sv = sparse_embed("transformer architecture")
    assert all(isinstance(i, int) for i in sv.indices)

def test_sparse_batch():
    from ingestion.sparse import sparse_embed_batch
    from qdrant_client import models
    results = sparse_embed_batch(["hello", "world"])
    assert len(results) == 2
    assert all(isinstance(r, models.SparseVector) for r in results)
```

- [ ] **Step 2: Run — expect FAIL**
```bash
cd ~/qdrant && python -m pytest tests/test_sparse.py -v
```

- [ ] **Step 3: Create `ingestion/sparse.py`**

```python
# ~/qdrant/ingestion/sparse.py
from typing import Optional
from fastembed import SparseTextEmbedding
from qdrant_client import models
import config

_model: Optional[SparseTextEmbedding] = None

def _get_model() -> SparseTextEmbedding:
    global _model
    if _model is None:
        _model = SparseTextEmbedding(model_name=config.BM25_MODEL)
    return _model

def sparse_embed(text: str) -> models.SparseVector:
    model = _get_model()
    embeddings = list(model.embed([text]))
    se = embeddings[0]
    return models.SparseVector(
        indices=se.indices.tolist(),
        values=se.values.tolist(),
    )

def sparse_embed_batch(texts: list[str]) -> list[models.SparseVector]:
    model = _get_model()
    results = []
    for se in model.embed(texts):
        results.append(models.SparseVector(
            indices=se.indices.tolist(),
            values=se.values.tolist(),
        ))
    return results
```

- [ ] **Step 4: Run tests — expect PASS**
```bash
cd ~/qdrant && python -m pytest tests/test_sparse.py -v
```

- [ ] **Step 5: Commit**
```bash
git add ingestion/sparse.py tests/test_sparse.py && \
git commit -m "feat: fastembed BM25 sparse vector wrapper"
```

---

### Task 8: `ingestion/chunker.py`

**Files:**
- Create: `~/qdrant/ingestion/chunker.py`
- Test: `~/qdrant/tests/test_chunker.py`

- [ ] **Step 1: Write failing tests**

```python
# ~/qdrant/tests/test_chunker.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

LONG_TEXT = "This is a sentence with some words. " * 200   # ~1400 tokens approx

def test_chunk_produces_child_and_parent():
    from ingestion.chunker import chunk_text
    chunks = chunk_text(LONG_TEXT)
    assert len(chunks) >= 2
    first = chunks[0]
    assert "child_text" in first
    assert "parent_text" in first
    assert "chunk_index" in first

def test_child_within_token_limit():
    from ingestion.chunker import chunk_text, count_tokens
    import config
    chunks = chunk_text(LONG_TEXT)
    for c in chunks:
        n = count_tokens(c["child_text"])
        assert n <= config.CHUNK_SIZE_TOKENS * 1.1, f"chunk too long: {n} tokens"

def test_parent_larger_than_child():
    from ingestion.chunker import chunk_text, count_tokens
    chunks = chunk_text(LONG_TEXT)
    for c in chunks:
        if c["parent_text"]:
            assert count_tokens(c["parent_text"]) >= count_tokens(c["child_text"])

def test_consecutive_indices():
    from ingestion.chunker import chunk_text
    chunks = chunk_text(LONG_TEXT)
    for i, c in enumerate(chunks):
        assert c["chunk_index"] == i

def test_short_text_single_chunk():
    from ingestion.chunker import chunk_text
    chunks = chunk_text("Short text.")
    assert len(chunks) == 1

def test_truncate_tokens():
    from ingestion.chunker import truncate_tokens, count_tokens
    text = "word " * 500
    truncated = truncate_tokens(text, 100)
    assert count_tokens(truncated) <= 100
```

- [ ] **Step 2: Run — expect FAIL**
```bash
cd ~/qdrant && python -m pytest tests/test_chunker.py -v
```

- [ ] **Step 3: Create `ingestion/chunker.py`**

```python
# ~/qdrant/ingestion/chunker.py
from typing import Optional
from functools import lru_cache
from transformers import AutoTokenizer
import config

@lru_cache(maxsize=1)
def _get_tokenizer():
    return AutoTokenizer.from_pretrained(
        config.EMBEDDING_TOKENIZER_ID,
        trust_remote_code=True,
    )

def count_tokens(text: str) -> int:
    tok = _get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))

def truncate_tokens(text: str, max_tokens: int) -> str:
    tok = _get_tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tok.decode(ids[:max_tokens])

def chunk_text(text: str) -> list[dict]:
    """
    Split text into child chunks with parent context windows.
    Returns list of dicts: {chunk_index, child_text, parent_text}
    """
    tok = _get_tokenizer()
    token_ids = tok.encode(text, add_special_tokens=False)
    total = len(token_ids)

    if total == 0:
        return []

    target  = config.CHUNK_SIZE_TOKENS
    overlap = int(target * config.CHUNK_OVERLAP_PCT)
    stride  = target - overlap

    # Build child chunk token spans
    spans = []
    start = 0
    while start < total:
        end = min(start + target, total)
        spans.append((start, end))
        if end == total:
            break
        start += stride

    parent_extra = int(target * config.PARENT_WINDOW_MULT / 2)

    chunks = []
    for idx, (cs, ce) in enumerate(spans):
        child_ids  = token_ids[cs:ce]
        ps         = max(0, cs - parent_extra)
        pe         = min(total, ce + parent_extra)
        parent_ids = token_ids[ps:pe]
        chunks.append({
            "chunk_index": idx,
            "child_text":  tok.decode(child_ids),
            "parent_text": tok.decode(parent_ids) if len(parent_ids) > len(child_ids) else None,
        })

    return chunks
```

- [ ] **Step 4: Run tests — expect PASS**
```bash
cd ~/qdrant && python -m pytest tests/test_chunker.py -v
```
Expected: All 6 PASS. Note: first run downloads the Qwen tokenizer (~few hundred MB).

- [ ] **Step 5: Commit**
```bash
git add ingestion/chunker.py tests/test_chunker.py && \
git commit -m "feat: Qwen-native tokenizer chunker with child/parent windows"
```

---

## Chunk 3: Parsers

### File Map

| File | Responsibility |
|---|---|
| `ingestion/parsers/__init__.py` | empty |
| `ingestion/parsers/base.py` | `Message`, `Conversation` dataclasses + `Parser` ABC |
| `ingestion/parsers/claude.py` | `ClaudeParser` |
| `ingestion/parsers/chatgpt.py` | `ChatGPTParser` |
| `ingestion/parsers/gemini.py` | `GeminiParser` |
| `tests/fixtures/claude_export.json` | Minimal anonymized Claude export |
| `tests/fixtures/chatgpt_export.json` | Minimal anonymized ChatGPT export |
| `tests/fixtures/gemini_export.json` | Minimal anonymized Gemini export |
| `tests/test_parsers.py` | Snapshot tests for all three parsers |

---

### Task 9: Parser base + fixtures + tests

- [ ] **Step 1: Create `ingestion/parsers/base.py`**

```python
# ~/qdrant/ingestion/parsers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass
class Message:
    role: str              # "user" | "assistant"
    content: str
    timestamp: Optional[str] = None   # ISO 8601 or None

@dataclass
class Conversation:
    platform: str          # "claude" | "chatgpt" | "gemini"
    conversation_id: str   # platform-native stable UUID/ID
    title: str
    date: str              # ISO 8601 date of first message
    messages: List[Message] = field(default_factory=list)

class Parser(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> List[Conversation]:
        ...

def detect_platform(data) -> Optional[str]:
    """Heuristic platform detection from parsed JSON."""
    if isinstance(data, list):
        # ChatGPT: top-level array with 'mapping' and 'conversation_id' per item
        if data and "mapping" in data[0] and "conversation_id" in data[0]:
            return "chatgpt"
    elif isinstance(data, dict):
        convos = data.get("conversations", [])
        if convos:
            first = convos[0] if isinstance(convos, list) else next(iter(convos.values()), {})
            if "uuid" in first and "chat_messages" in first:
                return "claude"
            if "id" in first and "reply" in first:
                return "gemini"
    return None
```

- [ ] **Step 2: Create minimal test fixtures**

```bash
mkdir -p ~/qdrant/tests/fixtures
```

Create `~/qdrant/tests/fixtures/claude_export.json`:
```json
{
  "conversations": [
    {
      "uuid": "aaaaaaaa-0001-0001-0001-000000000001",
      "name": "Test Conversation",
      "created_at": "2026-01-01T10:00:00Z",
      "chat_messages": [
        {"sender": "human", "text": "What is a transformer?", "created_at": "2026-01-01T10:00:00Z"},
        {"sender": "assistant", "text": "A transformer is a neural network architecture.", "created_at": "2026-01-01T10:00:01Z"}
      ]
    }
  ]
}
```

Create `~/qdrant/tests/fixtures/chatgpt_export.json`:
```json
[
  {
    "conversation_id": "bbbbbbbb-0001-0001-0001-000000000001",
    "title": "ChatGPT Test",
    "create_time": 1735725600.0,
    "mapping": {
      "node-1": {"message": {"author": {"role": "user"}, "content": {"parts": ["What is GPT?"]}, "create_time": 1735725600.0}, "children": ["node-2"]},
      "node-2": {"message": {"author": {"role": "assistant"}, "content": {"parts": ["GPT stands for Generative Pre-trained Transformer."]}, "create_time": 1735725601.0}, "children": []}
    }
  }
]
```

Create `~/qdrant/tests/fixtures/gemini_export.json`:
```json
{
  "conversations": [
    {
      "id": "cccccccc-0001-0001-0001-000000000001",
      "title": "Gemini Test",
      "reply": [
        {"author": "user", "text": "What is Gemini?", "timestamp": "2026-01-01T10:00:00Z"},
        {"author": "model", "text": "Gemini is Google's multimodal AI model.", "timestamp": "2026-01-01T10:00:01Z"}
      ]
    }
  ]
}
```

- [ ] **Step 3: Write snapshot tests**

```python
# ~/qdrant/tests/test_parsers.py
import json, sys, os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIXTURES = Path(__file__).parent / "fixtures"

def test_detect_platform_claude():
    from ingestion.parsers.base import detect_platform
    data = json.loads((FIXTURES / "claude_export.json").read_text())
    assert detect_platform(data) == "claude"

def test_detect_platform_chatgpt():
    from ingestion.parsers.base import detect_platform
    data = json.loads((FIXTURES / "chatgpt_export.json").read_text())
    assert detect_platform(data) == "chatgpt"

def test_detect_platform_gemini():
    from ingestion.parsers.base import detect_platform
    data = json.loads((FIXTURES / "gemini_export.json").read_text())
    assert detect_platform(data) == "gemini"

def test_claude_parser():
    from ingestion.parsers.claude import ClaudeParser
    convos = ClaudeParser().parse(FIXTURES / "claude_export.json")
    assert len(convos) == 1
    c = convos[0]
    assert c.platform == "claude"
    assert c.conversation_id == "aaaaaaaa-0001-0001-0001-000000000001"
    assert c.title == "Test Conversation"
    assert len(c.messages) == 2
    assert c.messages[0].role == "user"
    assert c.messages[1].role == "assistant"
    assert "transformer" in c.messages[0].content.lower()

def test_chatgpt_parser():
    from ingestion.parsers.chatgpt import ChatGPTParser
    convos = ChatGPTParser().parse(FIXTURES / "chatgpt_export.json")
    assert len(convos) == 1
    c = convos[0]
    assert c.platform == "chatgpt"
    assert c.conversation_id == "bbbbbbbb-0001-0001-0001-000000000001"
    assert len(c.messages) >= 2
    roles = {m.role for m in c.messages}
    assert "user" in roles and "assistant" in roles

def test_gemini_parser():
    from ingestion.parsers.gemini import GeminiParser
    convos = GeminiParser().parse(FIXTURES / "gemini_export.json")
    assert len(convos) == 1
    c = convos[0]
    assert c.platform == "gemini"
    assert c.conversation_id == "cccccccc-0001-0001-0001-000000000001"
    assert len(c.messages) == 2
```

- [ ] **Step 4: Run — expect FAIL (parsers not yet implemented)**
```bash
cd ~/qdrant && python -m pytest tests/test_parsers.py -v
```

- [ ] **Step 5: Create `ingestion/parsers/__init__.py`**
```bash
touch ~/qdrant/ingestion/parsers/__init__.py
```

- [ ] **Step 6: Create `ingestion/parsers/claude.py`**

```python
# ~/qdrant/ingestion/parsers/claude.py
import json
from pathlib import Path
from typing import List
from .base import Parser, Conversation, Message

class ClaudeParser(Parser):
    def parse(self, file_path: Path) -> List[Conversation]:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        raw = data.get("conversations", [])
        results = []
        for item in raw:
            conv_id = item.get("uuid", "")
            title   = item.get("name", "Untitled")
            msgs_raw = item.get("chat_messages", [])
            messages = []
            date = None
            for m in msgs_raw:
                role = "user" if m.get("sender") == "human" else "assistant"
                content = m.get("text", "") or ""
                ts = m.get("created_at")
                if date is None and ts:
                    date = ts[:10]
                messages.append(Message(role=role, content=content, timestamp=ts))
            results.append(Conversation(
                platform="claude",
                conversation_id=conv_id,
                title=title,
                date=date or "1970-01-01",
                messages=messages,
            ))
        return results
```

- [ ] **Step 7: Create `ingestion/parsers/chatgpt.py`**

```python
# ~/qdrant/ingestion/parsers/chatgpt.py
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from .base import Parser, Conversation, Message

class ChatGPTParser(Parser):
    def parse(self, file_path: Path) -> List[Conversation]:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        results = []
        for item in data:
            conv_id = item.get("conversation_id", "")
            title   = item.get("title", "Untitled")
            mapping = item.get("mapping", {})
            # Traverse mapping in order: find root → walk children
            messages = []
            date = None
            # Sort nodes by create_time
            nodes = sorted(
                [v for v in mapping.values() if v.get("message")],
                key=lambda n: n["message"].get("create_time") or 0,
            )
            for node in nodes:
                msg = node["message"]
                role_raw = msg.get("author", {}).get("role", "")
                if role_raw not in ("user", "assistant"):
                    continue
                parts   = msg.get("content", {}).get("parts", [])
                content = " ".join(str(p) for p in parts if isinstance(p, str))
                ts_raw  = msg.get("create_time")
                ts = None
                if ts_raw:
                    ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc).isoformat()
                    if date is None:
                        date = ts[:10]
                messages.append(Message(role=role_raw, content=content, timestamp=ts))
            results.append(Conversation(
                platform="chatgpt",
                conversation_id=conv_id,
                title=title,
                date=date or "1970-01-01",
                messages=messages,
            ))
        return results
```

- [ ] **Step 8: Create `ingestion/parsers/gemini.py`**

```python
# ~/qdrant/ingestion/parsers/gemini.py
import json
from pathlib import Path
from typing import List
from .base import Parser, Conversation, Message

class GeminiParser(Parser):
    def parse(self, file_path: Path) -> List[Conversation]:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        raw = data.get("conversations", [])
        results = []
        for item in raw:
            conv_id = item.get("id", "")
            title   = item.get("title", "Untitled")
            replies = item.get("reply", [])
            messages = []
            date = None
            for r in replies:
                author = r.get("author", "")
                role = "user" if author == "user" else "assistant"
                content = r.get("text", "") or ""
                ts = r.get("timestamp")
                if date is None and ts:
                    date = ts[:10]
                messages.append(Message(role=role, content=content, timestamp=ts))
            results.append(Conversation(
                platform="gemini",
                conversation_id=conv_id,
                title=title,
                date=date or "1970-01-01",
                messages=messages,
            ))
        return results
```

- [ ] **Step 9: Run tests — expect PASS**
```bash
cd ~/qdrant && python -m pytest tests/test_parsers.py -v
```
Expected: All 8 tests PASS.

- [ ] **Step 10: Commit**
```bash
git add ingestion/parsers/ tests/test_parsers.py tests/fixtures/ && \
git commit -m "feat: Claude/ChatGPT/Gemini parsers with snapshot tests"
```

---

## Chunk 4: Ingestion Scripts

### File Map

| File | Responsibility |
|---|---|
| `ingestion/ingest_docs.py` | Walk KNOWLEDGE_DIRS, chunk, embed, upsert |
| `ingestion/ingest_convos.py` | Parse exports, chunk, embed, upsert |

Both share helpers in `backend/db.py` (extended in this chunk).

---

### Task 10: Extend `backend/db.py` with source/chunk CRUD

- [ ] **Step 1: Add write helpers to `backend/db.py`**

Append to `backend/db.py`:

```python
import uuid as _uuid
import hashlib
import datetime
import config as _config

def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha256_normalize(text: str) -> str:
    """Whitespace-normalize (no lowercase) and SHA-256 hash."""
    import re
    normalized = re.sub(r'\s+', ' ', text.strip())
    return hashlib.sha256(normalized.encode()).hexdigest()

def make_point_id(source_id: int, chunk_index: int) -> str:
    return str(_uuid.uuid5(_config.NAMESPACE_BRAIN, f"{source_id}:{chunk_index}"))

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
    """Two-phase write: SQLite first (in transaction), then Qdrant."""
    from qdrant_client.models import PointStruct
    # Load category from sources table for payload
    src_row = conn.execute("SELECT * FROM sources WHERE id=?", [source_id]).fetchone()
    content_hash = sha256_normalize(chunk_text)
    point_id     = make_point_id(source_id, chunk_index)

    # Check if unchanged
    existing = conn.execute(
        "SELECT content_hash FROM chunks WHERE source_id=? AND chunk_index=?",
        [source_id, chunk_index]
    ).fetchone()
    if existing and existing["content_hash"] == content_hash:
        return  # no change

    # SQLite write + Qdrant upsert in one transaction window
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
              content_hash, _config.EMBEDDING_MODEL])

        sqlite_chunk_id = conn.execute(
            "SELECT id FROM chunks WHERE source_id=? AND chunk_index=?",
            [source_id, chunk_index]
        ).fetchone()["id"]

        # Get source metadata for payload
        src = conn.execute("SELECT * FROM sources WHERE id=?", [source_id]).fetchone()

        qdrant_client.upsert(
            collection_name=_config.QDRANT_COLLECTION,
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
                    "sqlite_chunk_id": sqlite_chunk_id,
                }
            )]
        )

def delete_source(conn, source_id: int, qdrant_client) -> None:
    """Delete all chunks for a source from Qdrant and SQLite."""
    from qdrant_client.models import PointIdsList
    point_ids = [r["qdrant_point_id"] for r in
                 conn.execute("SELECT qdrant_point_id FROM chunks WHERE source_id=?",
                              [source_id]).fetchall()]
    if point_ids:
        qdrant_client.delete(
            collection_name=_config.QDRANT_COLLECTION,
            points_selector=PointIdsList(points=point_ids),
        )
    conn.execute("DELETE FROM sources WHERE id=?", [source_id])
    conn.commit()
```

- [ ] **Step 2: Test the CRUD helpers**

Add to `tests/test_db.py`:

```python
def test_upsert_and_hash():
    import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from backend.db import get_connection, init_tables, sha256_normalize, make_point_id
    config.SQLITE_PATH = ":memory:"
    conn = get_connection()
    init_tables(conn)
    h1 = sha256_normalize("Hello World")
    h2 = sha256_normalize("Hello  World")   # extra space — should normalize same
    assert h1 == h2
    # case preserved
    h3 = sha256_normalize("foo()")
    h4 = sha256_normalize("FOO()")
    assert h3 != h4

def test_make_point_id_deterministic():
    from backend.db import make_point_id
    assert make_point_id(1, 0) == make_point_id(1, 0)
    assert make_point_id(1, 0) != make_point_id(1, 1)
```

```bash
cd ~/qdrant && python -m pytest tests/test_db.py -v
```
Expected: All tests PASS.

- [ ] **Step 3: Commit**
```bash
git add backend/db.py tests/test_db.py && \
git commit -m "feat: source/chunk CRUD with two-phase write and deterministic point IDs"
```

---

### Task 11: `ingestion/ingest_docs.py`

**Files:**
- Create: `~/qdrant/ingestion/ingest_docs.py`

- [ ] **Step 1: Create `ingest_docs.py`**

```python
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
from ingestion.chunker import chunk_text
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
```

- [ ] **Step 2: Smoke test (dry-run — no Qdrant writes)**
```bash
cd ~/qdrant && python ingestion/ingest_docs.py --dry-run 2>&1 | head -30
```
Expected: Lists files found, prints `[DRY] ... → N chunks`, no errors.

- [ ] **Step 3: Commit**
```bash
git add ingestion/ingest_docs.py && \
git commit -m "feat: ingest_docs.py with batch embed, two-phase write, and --sync safety gate"
```

---

### Task 12: `ingestion/ingest_convos.py`

**Files:**
- Create: `~/qdrant/ingestion/ingest_convos.py`

- [ ] **Step 1: Create `ingest_convos.py`**

```python
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
```

- [ ] **Step 2: Smoke test with fixture**
```bash
cd ~/qdrant && python ingestion/ingest_convos.py tests/fixtures/claude_export.json
```
Expected: `Parsed 1 conversations ... [1/1] Test Conversation`

- [ ] **Step 3: Commit**
```bash
git add ingestion/ingest_convos.py && \
git commit -m "feat: ingest_convos.py with per-platform parsing, conversation chunking, and checkpoint"
```

---

## Chunk 5: Watcher Daemon

### Task 13: `ingestion/watcher.py`

**Files:**
- Create: `~/qdrant/ingestion/watcher.py`

- [ ] **Step 1: Create `watcher.py`**

```python
#!/usr/bin/env python3
# ~/qdrant/ingestion/watcher.py
"""
Watch ~/Downloads for AI conversation export files and queue them for review.
Runs as a persistent daemon. Started by start.sh.
"""
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from backend.db import get_connection, init_tables, now_iso

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [watcher] %(message)s")
log = logging.getLogger(__name__)

def wait_for_file_complete(path: Path, stable_secs: float = 3.0,
                            timeout_secs: float = 60.0) -> bool:
    deadline = time.time() + timeout_secs
    last_size, stable_since = -1, None
    while time.time() < deadline:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        if size == last_size:
            if stable_since and (time.time() - stable_since) >= stable_secs:
                return True
        else:
            stable_since = time.time()
        last_size = size
        time.sleep(0.5)
    return False

def enqueue(path: Path, conn) -> None:
    known = conn.execute(
        "SELECT id FROM quarantine WHERE path=?", [str(path)]
    ).fetchone()
    if known:
        return
    if path.stat().st_size > config.MAX_FILE_SIZE_BYTES:
        log.warning(f"Skipping (too large): {path}")
        conn.execute(
            "INSERT INTO quarantine (path, detected_at, status, reason) VALUES (?,?,?,?)",
            [str(path), now_iso(), "rejected", "too_large"]
        )
        conn.commit()
        return
    conn.execute(
        "INSERT INTO quarantine (path, detected_at, status) VALUES (?,?,?)",
        [str(path), now_iso(), "pending"]
    )
    conn.commit()
    log.info(f"Queued for review: {path.name}")

def startup_scan(conn) -> None:
    """Enqueue any .json files in Downloads not already in quarantine."""
    watch_dir = Path(config.WATCHER_WATCH_DIR)
    if not watch_dir.exists():
        return
    known = {r["path"] for r in conn.execute("SELECT path FROM quarantine").fetchall()}
    for f in watch_dir.glob("*.json"):
        if str(f) not in known:
            log.info(f"Startup scan found: {f.name}")
            if wait_for_file_complete(f):
                enqueue(f, conn)

def process_approved(conn) -> None:
    """Trigger ingestion for any quarantine rows newly approved via the API."""
    import subprocess
    rows = conn.execute(
        "SELECT id, path FROM quarantine WHERE status='approved'"
    ).fetchall()
    for row in rows:
        log.info(f"Ingesting approved file: {row['path']}")
        try:
            subprocess.Popen(
                [sys.executable, str(Path(__file__).parent / "ingest_convos.py"), row["path"]],
                cwd=str(Path(__file__).parent.parent),
            )
            conn.execute(
                "UPDATE quarantine SET status='ingesting' WHERE id=?", [row["id"]]
            )
            conn.commit()
        except Exception as exc:
            log.error(f"Failed to start ingest: {exc}")

def main():
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    conn = get_connection()
    init_tables(conn)

    class Handler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            path = Path(event.src_path)
            if path.suffix.lower() not in config.WATCHER_ALLOWLIST:
                return
            log.info(f"Detected: {path.name} — waiting for complete...")
            if wait_for_file_complete(path):
                enqueue(path, conn)

    log.info(f"Starting watcher on {config.WATCHER_WATCH_DIR}")
    startup_scan(conn)

    observer = Observer()
    observer.schedule(Handler(), config.WATCHER_WATCH_DIR, recursive=False)
    observer.start()
    log.info("Watcher running. Press Ctrl+C to stop.")

    try:
        while True:
            process_approved(conn)
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test startup scan (no files in Downloads that match)**
```bash
cd ~/qdrant && timeout 3 python ingestion/watcher.py || true
```
Expected: `Starting watcher on .../Downloads`, `Watcher running.`, then exits after 3s — no errors.

- [ ] **Step 3: Commit**
```bash
git add ingestion/watcher.py && \
git commit -m "feat: watcher daemon with startup scan, file-complete check, and approve polling"
```

---

## Chunk 6: FastAPI Backend + Search

### File Map

| File | Responsibility |
|---|---|
| `backend/search.py` | Hybrid search: embed query → Qdrant RRF → fetch SQLite text → rerank |
| `backend/main.py` | FastAPI app: routes, reranker warmup, static file serving |

---

### Task 14: `backend/search.py`

**Files:**
- Create: `~/qdrant/backend/search.py`
- Test: `~/qdrant/tests/test_search.py`

- [ ] **Step 1: Write failing test**

```python
# ~/qdrant/tests/test_search.py
"""Integration test — requires Qdrant running with at least one ingested document."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_search_returns_list():
    from backend.search import search
    from backend.db import get_connection, init_tables
    conn = get_connection()
    init_tables(conn)
    results = search("attention mechanism", conn, limit=3)
    assert isinstance(results, list)
    # If Qdrant is empty, returns [] — that's fine, just not an error
    for r in results:
        assert "chunk_text" in r
        assert "score" in r
        assert "title" in r

def test_search_with_filters():
    from backend.search import search
    from backend.db import get_connection
    conn = get_connection()
    results = search("transformer", conn, platform="local", limit=5)
    assert isinstance(results, list)
```

- [ ] **Step 2: Run — expect FAIL**
```bash
cd ~/qdrant && python -m pytest tests/test_search.py -v
```

- [ ] **Step 3: Create `backend/search.py`**

```python
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
        from FlagEmbedding import FlagReranker
        _reranker = FlagReranker(
            config.RERANKER_MODEL,
            use_fp16=config.RERANKER_FP16,
            device=config.RERANKER_DEVICE,
        )
    return _reranker

def warmup_reranker():
    r = get_reranker()
    r.compute_score([["warmup", "warmup"]], normalize=True)

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
        scores = reranker.compute_score(pairs, normalize=True, batch_size=config.RERANK_BATCH_SIZE)
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
```

- [ ] **Step 4: Run tests**
```bash
cd ~/qdrant && python -m pytest tests/test_search.py -v
```
Expected: Both PASS (search returns empty list if no docs ingested — that's valid).

- [ ] **Step 5: Commit**
```bash
git add backend/search.py tests/test_search.py && \
git commit -m "feat: hybrid search with server-side RRF, filter push-down, and cross-encoder rerank"
```

---

### Task 15: `backend/main.py`

**Files:**
- Create: `~/qdrant/backend/main.py`

- [ ] **Step 1: Create `backend/main.py`**

```python
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
from ingestion.chunker import truncate_tokens

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
        "qdrant_vector_count": info.vectors_count,
        "qdrant_vector_on_disk": True,   # on_disk=True is set at collection creation
    }


@app.get("/quarantine")
def quarantine_list(status: Optional[str] = Query(None), conn=Depends(get_db)):
    q = "SELECT * FROM quarantine"
    if status:
        q += f" WHERE status='{status}'"
    q += " ORDER BY detected_at DESC"
    rows = [dict(r) for r in conn.execute(q).fetchall()]
    return {"items": rows}


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
    # Watcher polls for 'approved' rows and triggers ingest — no need to duplicate here
    return {"status": "approved", "path": row["path"]}


@app.post("/quarantine/{quarantine_id}/reject")
def quarantine_reject(quarantine_id: int, conn=Depends(get_db)):
    row = conn.execute("SELECT * FROM quarantine WHERE id=?", [quarantine_id]).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    conn.execute("UPDATE quarantine SET status='rejected' WHERE id=?", [quarantine_id])
    conn.commit()
    return {"status": "rejected"}
```

- [ ] **Step 2: Test backend starts**
```bash
cd ~/qdrant && uvicorn backend.main:app --workers 1 --port 8000 &
sleep 3
curl -s http://localhost:8000/status | python -m json.tool
kill %1
```
Expected: JSON with `total_chunks`, `total_sources`, etc. No crash.

- [ ] **Step 3: Commit**
```bash
git add backend/main.py && \
git commit -m "feat: FastAPI backend with search, status, quarantine endpoints, and reranker warmup"
```

---

## Chunk 7: React Frontend

### Task 16: React App

**Files:**
- Create: `~/qdrant/frontend/package.json`
- Create: `~/qdrant/frontend/vite.config.ts`
- Create: `~/qdrant/frontend/index.html`
- Create: `~/qdrant/frontend/src/types.ts`
- Create: `~/qdrant/frontend/src/main.tsx`
- Create: `~/qdrant/frontend/src/App.tsx`
- Create: `~/qdrant/frontend/src/components/SearchBar.tsx`
- Create: `~/qdrant/frontend/src/components/FilterPanel.tsx`
- Create: `~/qdrant/frontend/src/components/ResultsList.tsx`

- [ ] **Step 1: Scaffold React app**

```bash
cd ~/qdrant/frontend && npm create vite@latest . -- --template react-ts --yes
npm install
```

- [ ] **Step 2: Create `src/types.ts`**

```typescript
// ~/qdrant/frontend/src/types.ts
export interface SearchResult {
  score: number;
  title: string;
  platform: string;
  category: string;
  date: string;
  source_path: string;
  chunk_text: string;
  parent_text: string;
}

export interface SearchFilters {
  source_type: string;
  platform: string;
  category: string;
  date_from: string;
  date_to: string;
}
```

- [ ] **Step 3: Create `src/components/SearchBar.tsx`**

```tsx
// ~/qdrant/frontend/src/components/SearchBar.tsx
import { useState } from "react";

interface Props {
  onSearch: (query: string) => void;
  loading: boolean;
}

export function SearchBar({ onSearch, loading }: Props) {
  const [value, setValue] = useState("");

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (value.trim()) onSearch(value.trim());
  };

  return (
    <form onSubmit={submit} style={{ display: "flex", gap: 8 }}>
      <input
        value={value}
        onChange={e => setValue(e.target.value)}
        placeholder="Search your second brain..."
        style={{ flex: 1, padding: "10px 14px", fontSize: 16, borderRadius: 6,
                 border: "1px solid #ccc" }}
        disabled={loading}
      />
      <button type="submit" disabled={loading || !value.trim()}
        style={{ padding: "10px 20px", fontSize: 16, borderRadius: 6,
                 background: "#2563eb", color: "#fff", border: "none",
                 cursor: "pointer" }}>
        {loading ? "..." : "Search"}
      </button>
    </form>
  );
}
```

- [ ] **Step 4: Create `src/components/FilterPanel.tsx`**

```tsx
// ~/qdrant/frontend/src/components/FilterPanel.tsx
import { SearchFilters } from "../types";

interface Props {
  filters: SearchFilters;
  onChange: (f: SearchFilters) => void;
}

const PLATFORMS = ["", "local", "claude", "chatgpt", "gemini"];
const SOURCE_TYPES = ["", "document", "conversation"];

export function FilterPanel({ filters, onChange }: Props) {
  const set = (k: keyof SearchFilters, v: string) =>
    onChange({ ...filters, [k]: v });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, padding: 16,
                  background: "#f8f9fa", borderRadius: 8, border: "1px solid #e0e0e0" }}>
      <h3 style={{ margin: 0, fontSize: 14, color: "#555" }}>Filters</h3>

      <label style={{ fontSize: 13 }}>
        Source type
        <select value={filters.source_type} onChange={e => set("source_type", e.target.value)}
          style={{ display: "block", marginTop: 4, width: "100%", padding: "6px 8px" }}>
          {SOURCE_TYPES.map(t => <option key={t} value={t}>{t || "All"}</option>)}
        </select>
      </label>

      <label style={{ fontSize: 13 }}>
        Platform
        <select value={filters.platform} onChange={e => set("platform", e.target.value)}
          style={{ display: "block", marginTop: 4, width: "100%", padding: "6px 8px" }}>
          {PLATFORMS.map(p => <option key={p} value={p}>{p || "All"}</option>)}
        </select>
      </label>

      <label style={{ fontSize: 13 }}>
        Category
        <input value={filters.category} onChange={e => set("category", e.target.value)}
          placeholder="e.g. AI-Research"
          style={{ display: "block", marginTop: 4, width: "100%", padding: "6px 8px" }} />
      </label>

      <label style={{ fontSize: 13 }}>
        From date
        <input type="date" value={filters.date_from}
          onChange={e => set("date_from", e.target.value ? e.target.value + "T00:00:00Z" : "")}
          style={{ display: "block", marginTop: 4, width: "100%", padding: "6px 8px" }} />
      </label>

      <label style={{ fontSize: 13 }}>
        To date
        <input type="date" value={filters.date_to}
          onChange={e => set("date_to", e.target.value ? e.target.value + "T23:59:59Z" : "")}
          style={{ display: "block", marginTop: 4, width: "100%", padding: "6px 8px" }} />
      </label>
    </div>
  );
}
```

- [ ] **Step 5: Create `src/components/ResultsList.tsx`**

```tsx
// ~/qdrant/frontend/src/components/ResultsList.tsx
import { useState } from "react";
import { SearchResult } from "../types";

const PLATFORM_COLORS: Record<string, string> = {
  local: "#16a34a", claude: "#d97706", chatgpt: "#2563eb", gemini: "#7c3aed",
};

function ResultCard({ result }: { result: SearchResult }) {
  const [expanded, setExpanded] = useState(false);
  const color = PLATFORM_COLORS[result.platform] || "#555";
  const date = result.date ? result.date.slice(0, 10) : "";

  return (
    <div style={{ border: "1px solid #e0e0e0", borderRadius: 8, padding: 16, marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start",
                    marginBottom: 8 }}>
        <div>
          <span style={{ fontWeight: 600, fontSize: 15 }}>{result.title}</span>
          <span style={{ marginLeft: 10, background: color, color: "#fff", borderRadius: 4,
                         padding: "2px 8px", fontSize: 12 }}>{result.platform}</span>
          {result.category && (
            <span style={{ marginLeft: 6, background: "#e5e7eb", borderRadius: 4,
                           padding: "2px 8px", fontSize: 12 }}>{result.category}</span>
          )}
        </div>
        <span style={{ fontSize: 12, color: "#888" }}>
          {date} · {(result.score * 100).toFixed(0)}%
        </span>
      </div>

      <p style={{ margin: "0 0 8px", fontSize: 14, color: "#333", lineHeight: 1.5 }}>
        {result.chunk_text}
      </p>

      {result.parent_text && result.parent_text !== result.chunk_text && (
        <button onClick={() => setExpanded(!expanded)}
          style={{ fontSize: 12, color: "#2563eb", background: "none", border: "none",
                   cursor: "pointer", padding: 0 }}>
          {expanded ? "Hide context ▲" : "Show context ▼"}
        </button>
      )}
      {expanded && (
        <p style={{ margin: "8px 0 0", fontSize: 13, color: "#555",
                    background: "#f8f9fa", padding: 10, borderRadius: 4, lineHeight: 1.5 }}>
          {result.parent_text}
        </p>
      )}
    </div>
  );
}

interface Props {
  results: SearchResult[];
  loading: boolean;
  error: string | null;
  searched: boolean;
  onLoadMore: () => void;
  hasMore: boolean;
}

export function ResultsList({ results, loading, error, searched, onLoadMore, hasMore }: Props) {
  if (loading) return <div style={{ textAlign: "center", padding: 40, color: "#888" }}>Searching...</div>;
  if (error)   return <div style={{ color: "#dc2626", padding: 16 }}>Error: {error}</div>;
  if (searched && results.length === 0)
    return <div style={{ color: "#888", padding: 40, textAlign: "center" }}>
      No results found. Try broader filters or a different query.
    </div>;

  return (
    <div>
      {results.map((r, i) => <ResultCard key={i} result={r} />)}
      {hasMore && (
        <button onClick={onLoadMore}
          style={{ display: "block", margin: "0 auto", padding: "10px 24px",
                   background: "#f3f4f6", border: "1px solid #d1d5db", borderRadius: 6,
                   cursor: "pointer", fontSize: 14 }}>
          Load more
        </button>
      )}
    </div>
  );
}
```

- [ ] **Step 6: Create `src/App.tsx`**

```tsx
// ~/qdrant/frontend/src/App.tsx
import { useState } from "react";
import { SearchBar } from "./components/SearchBar";
import { FilterPanel } from "./components/FilterPanel";
import { ResultsList } from "./components/ResultsList";
import { SearchResult, SearchFilters } from "./types";

const PAGE_SIZE = 10;
const API_BASE  = "";   // same origin — FastAPI serves both API and static files

const EMPTY_FILTERS: SearchFilters = {
  source_type: "", platform: "", category: "", date_from: "", date_to: "",
};

export default function App() {
  const [query,    setQuery]    = useState("");
  const [filters,  setFilters]  = useState<SearchFilters>(EMPTY_FILTERS);
  const [results,  setResults]  = useState<SearchResult[]>([]);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState<string | null>(null);
  const [searched, setSearched] = useState(false);
  const [offset,   setOffset]   = useState(0);
  const [hasMore,  setHasMore]  = useState(false);

  const doSearch = async (q: string, off = 0, append = false) => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({ q, limit: String(PAGE_SIZE), offset: String(off) });
      if (filters.source_type) params.set("source_type", filters.source_type);
      if (filters.platform)    params.set("platform",    filters.platform);
      if (filters.category)    params.set("category",    filters.category);
      if (filters.date_from)   params.set("date_from",   filters.date_from);
      if (filters.date_to)     params.set("date_to",     filters.date_to);

      const res  = await fetch(`${API_BASE}/search?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      setResults(prev => append ? [...prev, ...data.results] : data.results);
      setHasMore(data.results.length === PAGE_SIZE);
      setOffset(off + data.results.length);
      setSearched(true);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (q: string) => {
    setQuery(q);
    setOffset(0);
    setResults([]);
    doSearch(q, 0, false);
  };

  const handleLoadMore = () => doSearch(query, offset, true);

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: 24, fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 4 }}>Second Brain</h1>
      <p style={{ color: "#888", marginBottom: 24, fontSize: 14 }}>
        Semantic search over your documents and AI conversations.
      </p>

      <SearchBar onSearch={handleSearch} loading={loading} />

      <div style={{ display: "grid", gridTemplateColumns: "260px 1fr", gap: 24, marginTop: 24 }}>
        <FilterPanel filters={filters} onChange={f => { setFilters(f); setResults([]); setSearched(false); }} />
        <ResultsList results={results} loading={loading} error={error}
                     searched={searched} onLoadMore={handleLoadMore} hasMore={hasMore} />
      </div>
    </div>
  );
}
```

- [ ] **Step 7: Update `src/main.tsx`**

```tsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

- [ ] **Step 8: Configure Vite proxy for dev (optional)**

In `vite.config.ts`, add proxy so `/search` hits the FastAPI backend during `npm run dev`:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/search':     'http://localhost:8000',
      '/status':     'http://localhost:8000',
      '/quarantine': 'http://localhost:8000',
    }
  }
})
```

- [ ] **Step 9: Build and verify**
```bash
cd ~/qdrant/frontend && npm run build
```
Expected: `dist/` folder created with `index.html` and assets.

- [ ] **Step 10: Test full stack**
```bash
cd ~/qdrant && uvicorn backend.main:app --workers 1 --port 8000 &
sleep 3
curl -s http://localhost:8000/status
# Open http://localhost:8000/app in browser
kill %1
```
Expected: Status JSON returned. Browser shows the Second Brain UI.

- [ ] **Step 11: Commit**
```bash
cd ~/qdrant && git add frontend/ && \
git commit -m "feat: React search UI with two-panel layout, filters, load-more, and context toggle"
```

---

## Final Integration Checklist

- [ ] Run `python init_schema.py` — collection created, meta written
- [ ] Run `python ingestion/ingest_docs.py --dry-run` — lists files, no errors
- [ ] Run `python ingestion/ingest_docs.py` on a small subset (one directory) — chunks appear in Qdrant
- [ ] Run `python ingestion/ingest_convos.py tests/fixtures/claude_export.json` — conversation ingested
- [ ] Start backend: `uvicorn backend.main:app --workers 1 --port 8000`
- [ ] `curl "http://localhost:8000/search?q=transformer"` — returns results
- [ ] Open `http://localhost:8000/app` — UI loads, search works
- [ ] Run `python ingestion/ingest_docs.py --sync` — no errors, 10% safety gate tested
- [ ] Run full test suite: `python -m pytest tests/ -v` — all pass
- [ ] Run `./start.sh` — all 5 processes start in order
- [ ] Run `./stop.sh` — all processes stop cleanly
