import logging
from pathlib import Path

import config
from qdrant_client.http.models import Distance
from whoosh.fields import BOOLEAN, ID, KEYWORD, TEXT, Schema

from data.etl.indexing_utils import (
    QdrantIndexSpec,
    WhooshIndexSpec,
    build_qdrant_index,
    build_whoosh_index,
)

def index_txt_bm25(jsonl_path: str, bm_index_path: str):
    """
    Build a Whoosh (BM25) index for TXT chunks (detex output).
    Expects fields like: chunk_id, text, paper_id, section_path, chunk_type, has_math_loss, harvest.
    """
    logger = logging.getLogger("etl")
    logger.info(
        "Starting index_txt_bm25 | jsonl_path=%s | bm_index_path=%s",
        jsonl_path,
        bm_index_path,
    )

    bm_index_path = Path(bm_index_path)
    schema = Schema(
        chunk_id=ID(stored=True, unique=True),
        text=TEXT(stored=False),
        paper_id=ID(stored=True),
        section=TEXT(stored=True),
        section_path=TEXT(stored=True),
        chunk_type=KEYWORD(stored=True, lowercase=True),
        labels=KEYWORD(stored=True, commas=True, lowercase=True),
        has_math_loss=BOOLEAN(stored=True),
        arxiv_ids=KEYWORD(stored=True, commas=True, lowercase=True),
        emails=KEYWORD(stored=True, commas=True, lowercase=True),
        urls=KEYWORD(stored=True, commas=True, lowercase=True),
    )

    def _doc_builder(rec, fallback_id: int):
        text = (rec.get("text") or "").strip()
        if not text:
            return None
        harvest = rec.get("harvest") or {}
        return {
            "chunk_id": str(rec.get("chunk_id", fallback_id)),
            "text": text,
            "paper_id": rec.get("paper_id", ""),
            "section": rec.get("section", rec.get("section_path", "")),
            "section_path": rec.get("section_path", ""),
            "chunk_type": (rec.get("chunk_type", "") or "").lower(),
            "labels": ",".join(rec.get("labels", [])),
            "has_math_loss": bool(rec.get("has_math_loss", False)),
            "arxiv_ids": ",".join(harvest.get("arxiv_ids", [])),
            "emails": ",".join(harvest.get("emails", [])),
            "urls": ",".join(harvest.get("urls", [])),
        }

    spec = WhooshIndexSpec(
        schema=schema,
        doc_builder=_doc_builder,
        description="plain-text chunks",
    )

    return build_whoosh_index(
        Path(jsonl_path),
        bm_index_path,
        spec=spec,
        logger=logger,
    )


def index_txt_qdrant(
    jsonl_path: str,
    qdrant_index_path: str,
    collection_name: str = config.TXT_QDRANT_COLLECTION,
    embedding_model: str = config.TXT_EMBEDDING_MODEL,
    batch_size: int = config.TXT_QDRANT_BATCH_SIZE,
):
    """
    Build a Qdrant dense index for TXT chunks. Stores full payloads for later retrieval.
    """
    logger = logging.getLogger("etl")
    logger.info(
        "Starting index_txt_qdrant | jsonl_path=%s | qdrant_index_path=%s | collection=%s | model=%s | batch_size=%s",
        jsonl_path,
        qdrant_index_path,
        collection_name,
        embedding_model,
        batch_size,
    )

    expected_dim = config.TXT_EMBEDDING_DIM

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
