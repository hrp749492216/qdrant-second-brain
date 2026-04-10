import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch
import numpy as np


@pytest.fixture
def mock_sparse_text_embedding():
    """Mock the SparseTextEmbedding model to avoid py-rust-stemmers segfault on Python 3.14"""
    with patch('ingestion.sparse.SparseTextEmbedding') as mock_model_class:
        mock_instance = MagicMock()
        mock_model_class.return_value = mock_instance

        # Create mock sparse embeddings
        mock_se1 = MagicMock()
        mock_se1.indices = np.array([0, 5, 12, 42], dtype=np.int32)
        mock_se1.values = np.array([0.8, 0.6, 0.95, 0.4], dtype=np.float32)

        mock_se2 = MagicMock()
        mock_se2.indices = np.array([1, 7], dtype=np.int32)
        mock_se2.values = np.array([0.5, 0.7], dtype=np.float32)

        mock_instance.embed.side_effect = lambda texts: iter([mock_se1, mock_se2][:len(texts)])

        yield mock_model_class


def test_sparse_returns_qdrant_type(mock_sparse_text_embedding):
    from ingestion.sparse import sparse_embed
    from qdrant_client import models
    sv = sparse_embed("hello world attention mechanism")
    assert isinstance(sv, models.SparseVector)
    assert len(sv.indices) > 0
    assert len(sv.indices) == len(sv.values)


def test_sparse_indices_are_ints(mock_sparse_text_embedding):
    from ingestion.sparse import sparse_embed
    sv = sparse_embed("transformer architecture")
    assert all(isinstance(i, int) for i in sv.indices)


def test_sparse_batch(mock_sparse_text_embedding):
    from ingestion.sparse import sparse_embed_batch
    from qdrant_client import models
    results = sparse_embed_batch(["hello", "world"])
    assert len(results) == 2
    assert all(isinstance(r, models.SparseVector) for r in results)
