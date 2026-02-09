import logging
from pathlib import Path

from etl.config import MD_QDRANT_COLLECTION, MD_EMBEDDING_MODEL, MD_QDRANT_BATCH_SIZE, MD_EMBEDDING_DIM
from qdrant_client.http.models import Distance
from whoosh.fields import ID, KEYWORD, NUMERIC, TEXT, Schema

from etl.indexing_utils import (
    QdrantIndexSpec,
    WhooshIndexSpec,
    build_qdrant_index,
    build_whoosh_index,
)


def index_md_bm25(json_path: str, bm_index_dir: str):
    """
    Build and store a BM25 (Whoosh) index from a markdown JSON file.

    Args:
        json_path: path to JSON file (dict keyed by chunk_id)
        bm_index_dir: directory where the Whoosh index will be saved

    Returns:
        dict summary with counts and output path
    """
    logger = logging.getLogger("etl")
    logger.info(
        "Starting index_md_bm25 | json_path=%s | bm_index_dir=%s",
        json_path,
        bm_index_dir,
    )

    json_path = Path(json_path)
    bm_index_dir = Path(bm_index_dir)
    bm_index_dir.mkdir(parents=True, exist_ok=True)

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
        Path(json_path),
        Path(bm_index_dir),
        spec=spec,
        logger=logger,
    )

def index_md_qdrant(
    json_path: str,
    collection_name: str = MD_QDRANT_COLLECTION,
    embedding_model: str = MD_EMBEDDING_MODEL,
    batch_size: int = MD_QDRANT_BATCH_SIZE,
):
    """
    Build and store a dense Qdrant vector index from a markdown JSON file.

    Args:
        json_path: path to JSON file (dict keyed by chunk_id)
        qdrant_index_path: directory for Qdrant local storage (e.g. "qdrant_storage/")
        collection_name: Qdrant collection name (default: config.MD_QDRANT_COLLECTION)
        embedding_model: embedding model name (default: config.MD_EMBEDDING_MODEL)
        batch_size: number of chunks to upsert per batch (default: config.MD_QDRANT_BATCH_SIZE)

    Returns:
        dict summary with counts and storage path
    """
    logger = logging.getLogger("etl")
    logger.info(
        "Starting index_md_qdrant | json_path=%s | collection=%s | model=%s | batch_size=%s",
        json_path,
        collection_name,
        embedding_model,
        batch_size,
    )

    expected_dim = MD_EMBEDDING_DIM

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
        Path(json_path),
        spec=spec,
        logger=logger,
    )
