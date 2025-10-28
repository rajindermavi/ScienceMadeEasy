import logging
from pathlib import Path

import config
from qdrant_client.http.models import Distance
from whoosh.fields import ID, KEYWORD, NUMERIC, TEXT, Schema

from data.etl.indexing_utils import (
    QdrantIndexSpec,
    WhooshIndexSpec,
    build_qdrant_index,
    build_whoosh_index,
)


def index_md_bm25(jsonl_path: str, bm_index_path: str):
    """
    Build and store a BM25 (Whoosh) index from a markdown JSONL file.

    Args:
        jsonl_path: path to JSONL file (each line = one chunk dict)
        bm_index_path: directory where the Whoosh index will be saved

    Returns:
        dict summary with counts and output path
    """
    logger = logging.getLogger("etl")
    logger.info(
        "Starting index_md_bm25 | jsonl_path=%s | bm_index_path=%s",
        jsonl_path,
        bm_index_path,
    )

    jsonl_path = Path(jsonl_path)
    bm_index_path = Path(bm_index_path)
    bm_index_path.mkdir(parents=True, exist_ok=True)

    schema = Schema(
        chunk_id=ID(stored=True, unique=True),
        text=TEXT(stored=False),
        section=TEXT(stored=True),
        labels=KEYWORD(stored=True, commas=True, lowercase=True),
        paper_id=ID(stored=True),
        year=NUMERIC(stored=True),
    )

    def _doc_builder(rec, fallback_id: int):
        text = (rec.get("text") or "").strip()
        if not text:
            return None
        return {
            "chunk_id": str(rec.get("chunk_id", fallback_id)),
            "text": text,
            "section": rec.get("section", ""),
            "labels": ",".join(rec.get("labels", [])),
            "paper_id": rec.get("paper_id", ""),
            "year": int(rec.get("year", 0)) if "year" in rec else 0,
        }

    spec = WhooshIndexSpec(
        schema=schema,
        doc_builder=_doc_builder,
        description="markdown chunks",
    )

    return build_whoosh_index(
        Path(jsonl_path),
        Path(bm_index_path),
        spec=spec,
        logger=logger,
    )

def index_md_qdrant(
    jsonl_path: str,
    qdrant_index_path: str,
    collection_name: str = config.MD_QDRANT_COLLECTION,
    embedding_model: str = config.MD_EMBEDDING_MODEL,
    batch_size: int = config.MD_QDRANT_BATCH_SIZE,
):
    """
    Build and store a dense Qdrant vector index from a markdown JSONL file.

    Args:
        jsonl_path: path to JSONL file (each line = one chunk dict)
        qdrant_index_path: directory for Qdrant local storage (e.g. "qdrant_storage/")
        collection_name: Qdrant collection name (default: config.MD_QDRANT_COLLECTION)
        embedding_model: embedding model name (default: config.MD_EMBEDDING_MODEL)
        batch_size: number of chunks to upsert per batch (default: config.MD_QDRANT_BATCH_SIZE)

    Returns:
        dict summary with counts and storage path
    """
    logger = logging.getLogger("etl")
    logger.info(
        "Starting index_md_qdrant | jsonl_path=%s | qdrant_index_path=%s | collection=%s | model=%s | batch_size=%s",
        jsonl_path,
        qdrant_index_path,
        collection_name,
        embedding_model,
        batch_size,
    )

    expected_dim = config.MD_EMBEDDING_DIM

    spec = QdrantIndexSpec(
        collection_name=collection_name,
        embedding_model=embedding_model,
        batch_size=batch_size,
        expected_dim=expected_dim,
        normalize_embeddings=True,
        distance=Distance.COSINE,
        payload_builder=lambda rec, _: rec,
    )

    return build_qdrant_index(
        Path(jsonl_path),
        Path(qdrant_index_path),
        spec=spec,
        logger=logger,
    )
