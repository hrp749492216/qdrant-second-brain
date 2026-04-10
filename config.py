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
