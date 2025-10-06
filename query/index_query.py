# hybrid_search_runtime.py

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from whoosh import index as bm25_index
from whoosh.qparser import MultifieldParser

from sentence_transformers import SentenceTransformer

import config
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    VectorParams,
)


# ---------- OPEN / LOAD ----------

def open_bm25_index(bm_index_path: str):
    """Reopen a persisted Whoosh index directory."""
    return bm25_index.open_dir(bm_index_path)

def connect_qdrant(
    qdrant_index_path: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    api_key: Optional[str] = None,
) -> QdrantClient:
    """
    Connect to Qdrant either in embedded mode (path=...) or server mode (host/port).
    Provide either qdrant_index_path OR (host[, port, api_key]).
    """
    if qdrant_index_path:
        return QdrantClient(path=str(Path(qdrant_index_path)))
    if host:
        return QdrantClient(host=host, port=port or 6333, api_key=api_key)
    raise ValueError("Provide qdrant_index_path for embedded OR host[/port] for server mode.")

def load_embed_model(name: str) -> SentenceTransformer:
    """Load the sentence-transformers embedding model."""
    return SentenceTransformer(name)


# ---------- SEARCHERS ----------

def bm25_search(ix, query: str, topk: int = 100) -> List[Tuple[str, float]]:
    """Return [(chunk_id, score)] from a Whoosh BM25 index."""
    with ix.searcher() as searcher:
        qp = MultifieldParser(["text", "section"], schema=ix.schema)
        q = qp.parse(query)
        rs = searcher.search(q, limit=topk)
        return [(r["chunk_id"], float(r.score)) for r in rs]

def dense_search(
    qdrant: QdrantClient,
    model: SentenceTransformer,
    query: str,
    collection_name: str,
    topk: int = 100
) -> List[Tuple[str, float]]:
    """Return [(chunk_id, score)] from Qdrant vector search (uses payload['chunk_id'])."""
    qv = model.encode(query, normalize_embeddings=True).tolist()
    hits = qdrant.search(collection_name=collection_name, query_vector=qv, limit=topk)
    return [(h.payload.get("chunk_id", str(h.id)), float(h.score)) for h in hits]

def rrf_fuse(*ranked_lists: List[Tuple[str, float]], k: int = 60) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion.
    Input: multiple ranked lists of (id, score).
    Output: list of (id, fused_score) sorted descending.
    """
    agg: Dict[str, float] = {}
    for lst in ranked_lists:
        for rank, (cid, _) in enumerate(lst, start=1):
            agg[cid] = agg.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(agg.items(), key=lambda x: x[1], reverse=True)


# ---------- PAYLOAD LOOKUP HELPERS ----------

def fetch_payloads_for_ids(
    qdrant: QdrantClient,
    collection_name: str,
    chunk_ids: List[str],
) -> Dict[str, Dict]:
    """
    Fetch payloads for given chunk_ids by filtering on payload field 'chunk_id'.
    Returns {chunk_id: payload_dict}.
    """
    out: Dict[str, Dict] = {}
    for cid in chunk_ids:
        flt = Filter(must=[FieldCondition(key="chunk_id", match=MatchValue(value=cid))])
        # scroll returns (points, next_page_offset). We only need the first match.
        points, _ = qdrant.scroll(collection_name=collection_name, scroll_filter=flt, limit=1)
        if points:
            out[cid] = points[0].payload
    return out


# ---------- TOP-LEVEL HYBRID SEARCH ----------

def hybrid_search_from_disk(
    query: str,
    bm_index_path: str,
    qdrant_index_path: Optional[str] = None,   # embedded
    qdrant_host: Optional[str] = None,         # server mode
    qdrant_port: Optional[int] = None,
    qdrant_api_key: Optional[str] = None,
    collection_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
    bm25_candidates: Optional[int] = None,
    dense_candidates: Optional[int] = None,
    topk: Optional[int] = None,
    expected_embedding_dim: Optional[int] = None,
    source: str = "md",
    return_payloads: bool = True,
) -> Dict:
    """
    Reopen stored BM25 + Qdrant indexes, run hybrid (BM25 + dense) with RRF fusion,
    and optionally return full payloads (text, section, metadata).

    The `source` argument toggles between default MD and TXT indexing settings.
    """
    settings = _get_source_defaults(source)

    ix = open_bm25_index(bm_index_path)
    qdrant = connect_qdrant(qdrant_index_path, qdrant_host, qdrant_port, qdrant_api_key)

    collection_name = collection_name or settings["collection"]
    embedding_model = embedding_model or settings["embedding_model"]
    bm25_candidates = bm25_candidates or settings["bm25_candidates"]
    dense_candidates = dense_candidates or settings["dense_candidates"]
    topk = topk or settings["topk"]
    expected_embedding_dim = (
        expected_embedding_dim
        if expected_embedding_dim is not None
        else settings["embedding_dim"]
    )

    model = load_embed_model(embedding_model)
    _ensure_embedding_dim_matches(
        qdrant,
        collection_name,
        embedding_model,
        model.get_sentence_embedding_dimension(),
        expected_embedding_dim,
    )

    bm_hits = bm25_search(ix, query, topk=bm25_candidates)
    de_hits = dense_search(qdrant, model, query, collection_name, topk=dense_candidates)

    bm_map = {cid: s for cid, s in bm_hits}
    de_map = {cid: s for cid, s in de_hits}

    fused = rrf_fuse(bm_hits, de_hits, k=60)
    top_ids = [cid for cid, _ in fused[:topk]]

    results = []
    payloads = fetch_payloads_for_ids(qdrant, collection_name, top_ids) if return_payloads else {}

    for cid in top_ids:
        item = {
            "chunk_id": cid,
            "fused_score": next(fs for (i, fs) in fused if i == cid),
            "bm25_score": bm_map.get(cid, 0.0),
            "dense_score": de_map.get(cid, 0.0),
        }
        if return_payloads and cid in payloads:
            p = payloads[cid]
            text = (p.get("text") or "").replace("\n", " ")
            item.update({
                "paper_id": p.get("paper_id"),
                "section": p.get("section"),
                "source_file": p.get("source_file"),
                "start_line": p.get("start_line"),
                "end_line": p.get("end_line"),
                "labels": p.get("labels"),
                "equations_raw": p.get("equations_raw"),
                "text_preview": text[:220] + ("…" if len(text) > 220 else ""),
            })
        results.append(item)

    return {
        "query": query,
        "results": results,
        "diagnostics": {
            "bm25_candidates": len(bm_hits),
            "dense_candidates": len(de_hits),
            "bm_index_path": str(Path(bm_index_path).resolve()),
            "qdrant_mode": "embedded" if qdrant_index_path else f"server:{qdrant_host}:{qdrant_port or 6333}",
            "collection_name": collection_name,
            "embedding_model": embedding_model,
            "source": source,
        }
    }


def _extract_collection_vector_size(qdrant: QdrantClient, collection_name: str) -> int:
    """Return the expected vector size for the target collection."""
    info = qdrant.get_collection(collection_name)
    vectors_cfg = info.config.params.vectors

    if isinstance(vectors_cfg, VectorParams):
        return vectors_cfg.size

    if isinstance(vectors_cfg, dict):
        sizes = {name: getattr(cfg, "size", None) for name, cfg in vectors_cfg.items()}
        unique_sizes = {s for s in sizes.values() if s is not None}
        if len(unique_sizes) == 1:
            return unique_sizes.pop()
        raise ValueError(
            "Collection contains multiple vector configurations; specify the vector name explicitly."
        )

    raise ValueError(
        f"Unable to determine vector size for collection '{collection_name}'."
    )


def _ensure_embedding_dim_matches(
    qdrant: QdrantClient,
    collection_name: str,
    embedding_model: str,
    model_dim: int,
    expected_dim: Optional[int],
):
    collection_dim = _extract_collection_vector_size(qdrant, collection_name)

    if expected_dim is not None and collection_dim != expected_dim:
        raise ValueError(
            "Embedding dimension mismatch: collection '%s' stores vectors of size %s "
            "but configured expectation is %s"
            % (collection_name, collection_dim, expected_dim)
        )

    if expected_dim is not None and model_dim != expected_dim:
        raise ValueError(
            "Embedding dimension mismatch: model '%s' outputs %s dims but config expects %s"
            % (embedding_model, model_dim, expected_dim)
        )

    if model_dim != collection_dim:
        raise ValueError(
            "Embedding dimension mismatch: collection '%s' stores vectors of size %s "
            "but model '%s' produces dimension %s."
            % (collection_name, collection_dim, embedding_model, model_dim)
        )


def _get_source_defaults(source: str) -> Dict[str, Optional[int]]:
    source_key = source.lower()
    if source_key == "md":
        return {
            "collection": config.MD_QDRANT_COLLECTION,
            "embedding_model": config.MD_EMBEDDING_MODEL,
            "embedding_dim": config.MD_EMBEDDING_DIM,
            "bm25_candidates": config.MD_BM25_CANDIDATES,
            "dense_candidates": config.MD_DENSE_CANDIDATES,
            "topk": config.MD_TOPK,
        }
    if source_key == "txt":
        return {
            "collection": config.TXT_QDRANT_COLLECTION,
            "embedding_model": config.TXT_EMBEDDING_MODEL,
            "embedding_dim": config.TXT_EMBEDDING_DIM,
            "bm25_candidates": config.TXT_BM25_CANDIDATES,
            "dense_candidates": config.TXT_DENSE_CANDIDATES,
            "topk": config.TXT_TOPK,
        }
    raise ValueError("Unsupported source '%s'. Expected 'md' or 'txt'." % source)


def hybrid_md_search_from_disk(*args, **kwargs):
    """Backward compatible alias for callers expecting the MD-specific function name."""
    kwargs.setdefault("source", "md")
    return hybrid_search_from_disk(*args, **kwargs)
