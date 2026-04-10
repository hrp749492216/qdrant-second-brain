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
