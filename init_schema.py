#!/usr/bin/env python3
# ~/qdrant/init_schema.py
"""
Idempotent setup: creates Qdrant collection + SQLite schema.
Run on every startup — safe to re-run.
"""
import sys
import config
from backend.db import get_connection, init_tables

def assert_meta_matches(conn) -> None:
    rows = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM meta")}
    stored_model = rows.get("embedding_model")
    stored_dim   = rows.get("embedding_dim")
    if stored_model is None:
        # First run or crashed before meta write — populate now
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('embedding_model', ?)", [config.EMBEDDING_MODEL])
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('embedding_dim', ?)",   [str(config.EMBEDDING_DIM_STORED)])
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '1')")
        conn.commit()
        print(f"[init_schema] meta written: model={config.EMBEDDING_MODEL} dim={config.EMBEDDING_DIM_STORED}")
        return
    assert stored_model == config.EMBEDDING_MODEL, (
        f"Embedding model mismatch: meta='{stored_model}' config='{config.EMBEDDING_MODEL}'. "
        "Run a migration or update config."
    )
    assert int(stored_dim) == config.EMBEDDING_DIM_STORED, (
        f"Embedding dim mismatch: meta={stored_dim} config={config.EMBEDDING_DIM_STORED}. "
        "Run a migration or update config."
    )
    print(f"[init_schema] meta OK: model={stored_model} dim={stored_dim}")


def setup_qdrant() -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        VectorParams, SparseVectorParams, Distance, HnswConfigDiff,
        ScalarQuantization, ScalarQuantizationConfig, ScalarType,
        Modifier, PayloadSchemaType,
    )
    client = QdrantClient(url=config.QDRANT_URL)
    existing = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config={
                "dense": VectorParams(
                    size=config.EMBEDDING_DIM_STORED,
                    distance=Distance.COSINE,
                    on_disk=True,
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(modifier=Modifier.IDF)
            },
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            ),
            on_disk_payload=False,
        )
        print(f"[init_schema] Qdrant collection '{config.QDRANT_COLLECTION}' created.")
    else:
        print(f"[init_schema] Qdrant collection '{config.QDRANT_COLLECTION}' already exists.")

    # Payload indexes — idempotent (ignore if already exists)
    for field, schema in [
        ("source_type",  PayloadSchemaType.KEYWORD),
        ("platform",     PayloadSchemaType.KEYWORD),
        ("category",     PayloadSchemaType.KEYWORD),
        ("date",         PayloadSchemaType.DATETIME),
        ("source_id",    PayloadSchemaType.INTEGER),
        ("content_hash", PayloadSchemaType.KEYWORD),
    ]:
        try:
            client.create_payload_index(config.QDRANT_COLLECTION,
                                        field_name=field, field_schema=schema)
        except Exception:
            pass


def main() -> None:
    print("[init_schema] Starting...")
    conn = get_connection()
    init_tables(conn)
    assert_meta_matches(conn)
    setup_qdrant()
    print("[init_schema] Done.")


if __name__ == "__main__":
    main()
