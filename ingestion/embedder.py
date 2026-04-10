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
