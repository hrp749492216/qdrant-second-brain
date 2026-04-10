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
