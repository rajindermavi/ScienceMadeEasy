import json
import logging
from pathlib import Path

import config
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from whoosh import index
from whoosh.fields import BOOLEAN, ID, KEYWORD, TEXT, Schema


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

    jsonl_path = Path(jsonl_path)
    bm_index_path = Path(bm_index_path)
    bm_index_path.mkdir(parents=True, exist_ok=True)

    schema = Schema(
        chunk_id=ID(stored=True, unique=True),
        text=TEXT(stored=False),
        paper_id=ID(stored=True),
        section_path=TEXT(stored=True),
        chunk_type=KEYWORD(stored=True, lowercase=True),
        labels=KEYWORD(stored=True, commas=True, lowercase=True),
        has_math_loss=BOOLEAN(stored=True),
        arxiv_ids=KEYWORD(stored=True, commas=True, lowercase=True),
        emails=KEYWORD(stored=True, commas=True, lowercase=True),
        urls=KEYWORD(stored=True, commas=True, lowercase=True),
    )

    if index.exists_in(bm_index_path):
        logger.info("Opening existing TXT BM25 index at %s", bm_index_path)
        ix = index.open_dir(bm_index_path)
        writer = ix.writer()
    else:
        logger.info("Creating new TXT BM25 index at %s", bm_index_path)
        ix = index.create_in(bm_index_path, schema)
        writer = ix.writer()

    total = 0
    skipped_blank = 0
    skipped_decode = 0
    skipped_empty_text = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                skipped_blank += 1
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                skipped_decode += 1
                logger.warning(
                    "index_txt_bm25: skipping line %s due to JSON decode error: %s",
                    line_no,
                    exc,
                )
                continue
            text = (rec.get("text") or "").strip()
            if not text:
                skipped_empty_text += 1
                continue
            harvest = rec.get("harvest") or {}
            writer.add_document(
                chunk_id=str(rec.get("chunk_id", total)),
                text=text,
                paper_id=rec.get("paper_id", ""),
                section_path=rec.get("section_path", ""),
                chunk_type=(rec.get("chunk_type", "") or "").lower(),
                labels=",".join(rec.get("labels", [])),
                has_math_loss=bool(rec.get("has_math_loss", False)),
                arxiv_ids=",".join(harvest.get("arxiv_ids", [])),
                emails=",".join(harvest.get("emails", [])),
                urls=",".join(harvest.get("urls", [])),
            )
            total += 1

    writer.commit()
    logger.info(
        "index_txt_bm25 complete | indexed=%s | skipped_blank=%s | skipped_decode=%s | skipped_empty_text=%s",
        total,
        skipped_blank,
        skipped_decode,
        skipped_empty_text,
    )

    return {
        "records_indexed": total,
        "index_path": str(bm_index_path.resolve())
    }


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

    jsonl_path = Path(jsonl_path)
    qdrant_index_path = Path(qdrant_index_path)
    qdrant_index_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading embedding model %s", embedding_model)
    model = SentenceTransformer(embedding_model)
    dim = model.get_sentence_embedding_dimension()
    logger.info("Embedding dimension resolved to %s", dim)

    expected_dim = config.TXT_EMBEDDING_DIM
    if expected_dim is not None and dim != expected_dim:
        raise ValueError(
            "Embedding dimension mismatch: model '%s' outputs %s dims but config expects %s"
            % (embedding_model, dim, expected_dim)
        )

    qdrant = QdrantClient(path=str(qdrant_index_path))

    # Fresh collection each time; switch to create_collection + upsert if you prefer incremental
    existing_collections = {c.name for c in qdrant.get_collections().collections}
    if collection_name in existing_collections:
        logger.info("Deleting existing TXT Qdrant collection %s", collection_name)
        qdrant.delete_collection(collection_name)
    logger.info("Recreating TXT Qdrant collection %s", collection_name)
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    logger.info("\tRecreation complete")

    points = []
    total = 0
    batch_counter = 0
    skipped_blank = 0
    skipped_decode = 0
    skipped_empty_text = 0
    progress_interval = max(1, config.TXT_QDRANT_PROGRESS_INTERVAL)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                skipped_blank += 1
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                skipped_decode += 1
                logger.warning(
                    "index_txt_qdrant: skipping line %s due to JSON decode error: %s",
                    line_no,
                    exc,
                )
                continue
            text = (rec.get("text") or "").strip()
            if not text:
                skipped_empty_text += 1
                continue
            vec = model.encode(text, normalize_embeddings=True).tolist()
            points.append(PointStruct(id=total, vector=vec, payload=rec))
            total += 1
            if len(points) >= batch_size:
                logger.info(
                    "Upserting batch of %s TXT points into Qdrant collection %s",
                    len(points),
                    collection_name,
                )
                qdrant.upsert(collection_name=collection_name, points=points)
                points = []
                batch_counter += 1
                if batch_counter % progress_interval == 0:
                    logger.info(
                        "index_txt_qdrant progress | batches=%s | total_points=%s",
                        batch_counter,
                        total,
                    )

    if points:
        logger.info(
            "Upserting final batch of %s TXT points into Qdrant collection %s",
            len(points),
            collection_name,
        )
        qdrant.upsert(collection_name=collection_name, points=points)
        batch_counter += 1

    if batch_counter and batch_counter % progress_interval != 0:
        logger.info(
            "index_txt_qdrant progress | batches=%s | total_points=%s",
            batch_counter,
            total,
        )

    logger.info(
        "index_txt_qdrant complete | indexed=%s | skipped_blank=%s | skipped_decode=%s | skipped_empty_text=%s",
        total,
        skipped_blank,
        skipped_decode,
        skipped_empty_text,
    )

    return {
        "records_indexed": total,
        "index_path": str(qdrant_index_path.resolve()),
        "collection_name": collection_name,
        "embedding_model": embedding_model,
    }
