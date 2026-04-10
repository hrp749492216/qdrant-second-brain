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
