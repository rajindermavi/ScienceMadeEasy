import json
import logging
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from whoosh import index
from whoosh.fields import ID, KEYWORD, NUMERIC, TEXT, Schema


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

    # Define schema
    schema = Schema(
        chunk_id=ID(stored=True, unique=True),
        text=TEXT(stored=False),           # main text for BM25 ranking
        section=TEXT(stored=True),
        labels=KEYWORD(stored=True, commas=True, lowercase=True),
        paper_id=ID(stored=True),
        year=NUMERIC(stored=True)
    )

    # Create or overwrite index
    if index.exists_in(bm_index_path):
        logger.info("Opening existing BM25 index at %s", bm_index_path)
        ix = index.open_dir(bm_index_path)
        writer = ix.writer()
    else:
        logger.info("Creating new BM25 index at %s", bm_index_path)
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
                    "index_md_bm25: skipping line %s due to JSON decode error: %s",
                    line_no,
                    exc,
                )
                continue
            text = rec.get("text", "").strip()
            if not text:
                skipped_empty_text += 1
                continue

            writer.add_document(
                chunk_id=str(rec.get("chunk_id", total)),
                text=text,
                section=rec.get("section", ""),
                labels=",".join(rec.get("labels", [])),
                paper_id=rec.get("paper_id", ""),
                year=int(rec.get("year", 0)) if "year" in rec else 0
            )
            total += 1

    writer.commit()
    logger.info(
        "index_md_bm25 complete | indexed=%s | skipped_blank=%s | skipped_decode=%s | skipped_empty_text=%s",
        total,
        skipped_blank,
        skipped_decode,
        skipped_empty_text,
    )

    return {
        "records_indexed": total,
        "index_path": str(bm_index_path.resolve())
    }

def index_md_qdrant(
    jsonl_path: str,
    qdrant_index_path: str,
    collection_name: str = "md_chunks",
    embedding_model: str = "BAAI/bge-small-en",
    batch_size: int = 64,
):
    """
    Build and store a dense Qdrant vector index from a markdown JSONL file.

    Args:
        jsonl_path: path to JSONL file (each line = one chunk dict)
        qdrant_index_path: directory for Qdrant local storage (e.g. "qdrant_storage/")
        collection_name: Qdrant collection name (default: "md_chunks")
        embedding_model: embedding model name (default: bge-base-en-v1.5)
        batch_size: number of chunks to upsert per batch

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

    jsonl_path = Path(jsonl_path)
    qdrant_index_path = Path(qdrant_index_path)
    qdrant_index_path.mkdir(parents=True, exist_ok=True)

    # --- Load embedding model
    logger.info("Loading embedding model %s", embedding_model)
    model = SentenceTransformer(embedding_model)
    dim = model.get_sentence_embedding_dimension()
    logger.info("Embedding dimension resolved to %s", dim)

    # --- Connect to embedded Qdrant (no external server required)
    qdrant = QdrantClient(path=str(qdrant_index_path))

    # --- Create (or recreate) collection
    existing_collections = {c.name for c in qdrant.get_collections().collections}
    if collection_name in existing_collections:
        logger.info("Deleting existing Qdrant collection %s", collection_name)
        qdrant.delete_collection(collection_name)
    logger.info("Recreating Qdrant collection %s", collection_name)
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    logger.info("\tRecreation complete")

    total = 0
    points = []
    batch_counter = 0
    skipped_blank = 0
    skipped_decode = 0
    skipped_empty_text = 0

    # --- Ingest and embed in batches
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
                    "index_md_qdrant: skipping line %s due to JSON decode error: %s",
                    line_no,
                    exc,
                )
                continue
            text = rec.get("text", "").strip()
            if not text:
                skipped_empty_text += 1
                continue

            vec = model.encode(text, normalize_embeddings=True).tolist()
            point = PointStruct(
                id=total,
                vector=vec,
                payload=rec,  # store the full record for later retrieval
            )
            points.append(point)
            total += 1

            if len(points) >= batch_size:
                logger.info(
                    "Upserting batch of %s points into Qdrant collection %s",
                    len(points),
                    collection_name,
                )
                qdrant.upsert(collection_name=collection_name, points=points)
                points = []
                batch_counter += 1
                if batch_counter % 64 == 0:
                    logger.info(
                        "index_md_qdrant progress | batches=%s | total_points=%s",
                        batch_counter,
                        total,
                    )

    if points:
        logger.info(
            "Upserting final batch of %s points into Qdrant collection %s",
            len(points),
            collection_name,
        )
        qdrant.upsert(collection_name=collection_name, points=points)
        batch_counter += 1

    if batch_counter and batch_counter % 64 != 0:
        logger.info(
            "index_md_qdrant progress | batches=%s | total_points=%s",
            batch_counter,
            total,
        )

    logger.info(
        "index_md_qdrant complete | indexed=%s | skipped_blank=%s | skipped_decode=%s | skipped_empty_text=%s",
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
