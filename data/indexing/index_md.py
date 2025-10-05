import json
from pathlib import Path
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, KEYWORD, NUMERIC
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


def index_md_bm25(jsonl_path: str, bm_index_path: str):
    """
    Build and store a BM25 (Whoosh) index from a markdown JSONL file.

    Args:
        jsonl_path: path to JSONL file (each line = one chunk dict)
        bm_index_path: directory where the Whoosh index will be saved

    Returns:
        dict summary with counts and output path
    """
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
        ix = index.open_dir(bm_index_path)
        writer = ix.writer()
    else:
        ix = index.create_in(bm_index_path, schema)
        writer = ix.writer()

    total = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("text", "").strip()
            if not text:
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
    return {
        "records_indexed": total,
        "index_path": str(bm_index_path.resolve())
    }

def index_md_qdrant(jsonl_path: str, qdrant_index_path: str,
                    collection_name: str = "md_chunks",
                    embedding_model: str = "BAAI/bge-base-en-v1.5",
                    batch_size: int = 256):
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
    jsonl_path = Path(jsonl_path)
    qdrant_index_path = Path(qdrant_index_path)
    qdrant_index_path.mkdir(parents=True, exist_ok=True)

    # --- Load embedding model
    model = SentenceTransformer(embedding_model)
    dim = model.get_sentence_embedding_dimension()

    # --- Connect to embedded Qdrant (no external server required)
    qdrant = QdrantClient(path=str(qdrant_index_path))

    # --- Create (or recreate) collection
    if collection_name in [c.name for c in qdrant.get_collections().collections]:
        qdrant.delete_collection(collection_name)
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    total = 0
    points = []

    # --- Ingest and embed in batches
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("text", "").strip()
            if not text:
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
                qdrant.upsert(collection_name=collection_name, points=points)
                points = []

    if points:
        qdrant.upsert(collection_name=collection_name, points=points)

    return {
        "records_indexed": total,
        "index_path": str(qdrant_index_path.resolve()),
        "collection_name": collection_name,
        "embedding_model": embedding_model,
    }