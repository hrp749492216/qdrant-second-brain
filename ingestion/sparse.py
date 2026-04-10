# ~/qdrant/ingestion/sparse.py
# BM25-style TF scoring with feature hashing for stable vocabulary.
# fastembed SparseTextEmbedding (Qdrant/bm25) segfaults on Python 3.14 due to
# py-rust-stemmers native incompatibility. This pure-Python implementation
# produces BM25 TF scores; Qdrant's Modifier.IDF handles corpus IDF at query time.
import hashlib
import re
from collections import Counter
from qdrant_client import models

_VOCAB_SIZE = 65536   # hash space — large enough to avoid frequent collisions


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())


def _token_index(token: str) -> int:
    """Deterministic mapping: token → [0, VOCAB_SIZE)."""
    return int(hashlib.md5(token.encode(), usedforsecurity=False).hexdigest(), 16) % _VOCAB_SIZE


def sparse_embed(text: str, k1: float = 1.5) -> models.SparseVector:
    """BM25 TF saturation — Qdrant multiplies by IDF via Modifier.IDF."""
    tokens = _tokenize(text)
    if not tokens:
        return models.SparseVector(indices=[0], values=[0.001])

    tf = Counter(tokens)
    indices: list[int] = []
    values: list[float] = []
    seen: dict[int, int] = {}

    for token, count in tf.items():
        idx = _token_index(token)
        # BM25 TF saturation formula (length normalisation omitted — Qdrant IDF handles it)
        bm25_tf = float((count * (k1 + 1)) / (count + k1))
        if idx in seen:
            values[seen[idx]] = max(values[seen[idx]], bm25_tf)
        else:
            seen[idx] = len(indices)
            indices.append(idx)
            values.append(bm25_tf)

    return models.SparseVector(indices=indices, values=values)


def sparse_embed_batch(texts: list[str]) -> list[models.SparseVector]:
    return [sparse_embed(t) for t in texts]
