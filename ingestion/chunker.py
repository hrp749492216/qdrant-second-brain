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
