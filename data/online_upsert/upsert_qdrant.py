# upsert_qdrant.py
import json, hashlib
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

import config

def _stable_int_id(chunk_id: str) -> int:
    # 63-bit int from sha1 (fits qdrant int id)
    h = hashlib.sha1(chunk_id.encode("utf-8")).hexdigest()
    return int(h[:15], 16)

def ensure_collection(qdrant: QdrantClient, collection: str, dim: int):
    cols = [c.name for c in qdrant.get_collections().collections]
    if collection not in cols:
        qdrant.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

def upsert_md_qdrant_from_jsonl(
    jsonl_path: str,
    qdrant_path: str = None,  # embedded
    host: str = None, port: int = 6333,  # server
    collection_name: str = "md_chunks",
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 128
) -> int:
    # connect
    qdrant = QdrantClient(path=qdrant_path) if qdrant_path else QdrantClient(host=host, port=port)
    model = SentenceTransformer(embedding_model)
    dim = model.get_sentence_embedding_dimension()
    ensure_collection(qdrant, collection_name, dim)

    pts, total = [], 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            text = (rec.get("text") or "").strip()
            if not text: continue
            vec = model.encode(text, normalize_embeddings=True).tolist()
            pid = _stable_int_id(str(rec["chunk_id"]))
            pts.append(PointStruct(id=pid, vector=vec, payload=rec))
            if len(pts) >= batch_size:
                qdrant.upsert(collection_name=collection_name, points=pts); pts = []
            total += 1
    if pts:
        qdrant.upsert(collection_name=collection_name, points=pts)
    return total
