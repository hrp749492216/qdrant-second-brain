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


def test_sparse_deterministic():
    """Same text must always produce same indices and values."""
    from ingestion.sparse import sparse_embed
    sv1 = sparse_embed("deterministic test sentence")
    sv2 = sparse_embed("deterministic test sentence")
    assert sv1.indices == sv2.indices
    assert sv1.values == sv2.values


def test_sparse_different_texts_differ():
    from ingestion.sparse import sparse_embed
    sv1 = sparse_embed("apple orange banana")
    sv2 = sparse_embed("neural network transformer")
    assert set(sv1.indices) != set(sv2.indices)
