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
