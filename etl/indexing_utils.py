from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from whoosh import index
from whoosh.fields import Schema

JsonDict = Dict[str, Any]


def _load_jsonl(path: Path) -> Iterator[JsonDict]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                yield {"__skip__": "blank", "__line__": line_no}
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                yield {"__skip__": "decode_error", "__line__": line_no, "__error__": exc}


@dataclass
class WhooshIndexSpec:
    """Configuration for building a Whoosh index."""

    schema: Schema
    doc_builder: Callable[[JsonDict, int], Optional[JsonDict]]
    description: str = "records"
    mode: str = "overwrite"  # or "append"


def build_whoosh_index(
    jsonl_path: Path,
    index_dir: Path,
    spec: WhooshIndexSpec,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Build (or append to) a Whoosh index based on a specification.
    """
    logger = logger or logging.getLogger("etl")
    index_dir.mkdir(parents=True, exist_ok=True)

    if index.exists_in(index_dir):
        ix = index.open_dir(index_dir)
        writer = ix.writer()
        logger.info("Opening existing Whoosh index at %s", index_dir)
    else:
        ix = index.create_in(index_dir, spec.schema)
        writer = ix.writer()
        logger.info("Creating new Whoosh index at %s", index_dir)

    totals = {"indexed": 0, "skipped_blank": 0, "skipped_decode": 0, "skipped_filter": 0}

    for payload in _load_jsonl(jsonl_path):
        skip_code = payload.get("__skip__")
        if skip_code == "blank":
            totals["skipped_blank"] += 1
            continue
        if skip_code == "decode_error":
            totals["skipped_decode"] += 1
            logger.warning(
                "build_whoosh_index: skipping line %s due to JSON decode error: %s",
                payload.get("__line__"),
                payload.get("__error__"),
            )
            continue

        doc = spec.doc_builder(payload, totals["indexed"])
        if not doc:
            totals["skipped_filter"] += 1
            continue
        writer.add_document(**doc)
        totals["indexed"] += 1

    writer.commit()
    logger.info(
        "Whoosh indexing complete | indexed=%s | skipped_blank=%s | skipped_decode=%s | skipped_filter=%s",
        totals["indexed"],
        totals["skipped_blank"],
        totals["skipped_decode"],
        totals["skipped_filter"],
    )
    return {
        "records_indexed": totals["indexed"],
        "skipped_blank": totals["skipped_blank"],
        "skipped_decode": totals["skipped_decode"],
        "skipped_filtered": totals["skipped_filter"],
        "index_path": str(index_dir.resolve()),
        "description": spec.description,
    }


@dataclass
class QdrantIndexSpec:
    """Configuration for building a Qdrant index."""

    collection_name: str
    embedding_model: str
    batch_size: int
    expected_dim: Optional[int] = None
    normalize_embeddings: bool = True
    payload_builder: Callable[[JsonDict, int], Optional[JsonDict]] = lambda rec, _: rec
    text_getter: Callable[[JsonDict], str] = lambda rec: str(rec.get("text", "") or "")
    distance: Distance = Distance.COSINE
    recreate_collection: bool = True


def build_qdrant_index(
    jsonl_path: Path,
    storage_dir: Path,
    spec: QdrantIndexSpec,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Build (or rebuild) a Qdrant local collection from JSONL payloads.
    """
    logger = logger or logging.getLogger("etl")
    storage_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading embedding model %s", spec.embedding_model)
    model = SentenceTransformer(spec.embedding_model)
    dim = model.get_sentence_embedding_dimension()
    if spec.expected_dim is not None and dim != spec.expected_dim:
        raise ValueError(
            "Embedding dimension mismatch: model '%s' returns %s dims but spec expects %s"
            % (spec.embedding_model, dim, spec.expected_dim)
        )

    client = QdrantClient(path=str(storage_dir))
    existing = {c.name for c in client.get_collections().collections}
    if spec.collection_name in existing and spec.recreate_collection:
        logger.info("Deleting existing Qdrant collection %s", spec.collection_name)
        client.delete_collection(spec.collection_name)

    if spec.recreate_collection or spec.collection_name not in existing:
        logger.info("Recreating Qdrant collection %s", spec.collection_name)
        client.recreate_collection(
            collection_name=spec.collection_name,
            vectors_config=VectorParams(size=dim, distance=spec.distance),
        )

    totals = {"indexed": 0, "skipped_blank": 0, "skipped_decode": 0, "skipped_empty_text": 0}
    batch: list[PointStruct] = []
    batch_idx = 0

    for payload in _load_jsonl(jsonl_path):
        skip_code = payload.get("__skip__")
        if skip_code == "blank":
            totals["skipped_blank"] += 1
            continue
        if skip_code == "decode_error":
            totals["skipped_decode"] += 1
            logger.warning(
                "build_qdrant_index: skipping line %s due to JSON decode error: %s",
                payload.get("__line__"),
                payload.get("__error__"),
            )
            continue

        text = spec.text_getter(payload).strip()
        if not text:
            totals["skipped_empty_text"] += 1
            continue

        vector = model.encode(text, normalize_embeddings=spec.normalize_embeddings).tolist()
        point_payload = spec.payload_builder(payload, totals["indexed"])
        if point_payload is None:
            totals["skipped_empty_text"] += 1
            continue

        batch.append(PointStruct(id=totals["indexed"], vector=vector, payload=point_payload))
        totals["indexed"] += 1

        if len(batch) >= spec.batch_size:
            _upsert_batch(client, spec.collection_name, batch, logger)
            batch = []
            batch_idx += 1

    if batch:
        _upsert_batch(client, spec.collection_name, batch, logger)
        batch_idx += 1

    logger.info(
        "Qdrant indexing complete | indexed=%s | batches=%s | skipped_blank=%s | skipped_decode=%s | skipped_empty_text=%s",
        totals["indexed"],
        batch_idx,
        totals["skipped_blank"],
        totals["skipped_decode"],
        totals["skipped_empty_text"],
    )

    return {
        "records_indexed": totals["indexed"],
        "skipped_blank": totals["skipped_blank"],
        "skipped_decode": totals["skipped_decode"],
        "skipped_empty_text": totals["skipped_empty_text"],
        "collection_name": spec.collection_name,
        "embedding_model": spec.embedding_model,
        "index_path": str(storage_dir.resolve()),
    }


def _upsert_batch(client: QdrantClient, collection: str, points: Iterable[PointStruct], logger: logging.Logger) -> None:
    points = list(points)
    if not points:
        return
    logger.info("Upserting batch of %s points into Qdrant collection %s", len(points), collection)
    client.upsert(collection_name=collection, points=points)

