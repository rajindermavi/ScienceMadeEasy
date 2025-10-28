from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sentence_transformers import CrossEncoder, SentenceTransformer
from whoosh import index as bm25_index
from whoosh.qparser import MultifieldParser

import config
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue, VectorParams


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


@lru_cache(maxsize=4)
def load_embed_model(name: str) -> SentenceTransformer:
    """Load the sentence-transformers embedding model with caching."""
    return SentenceTransformer(name)


@lru_cache(maxsize=2)
def _load_reranker(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


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
    topk: int = 100,
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


# ---------- SETTINGS & SERVICE LAYER ----------


@dataclass(frozen=True)
class HybridSearchSettings:
    source: str
    bm_index_path: Path
    qdrant_index_path: Optional[Path]
    qdrant_host: Optional[str]
    qdrant_port: Optional[int]
    qdrant_api_key: Optional[str]
    collection_name: str
    embedding_model: str
    embedding_dim: Optional[int]
    bm25_candidates: int
    dense_candidates: int
    topk: int

    @classmethod
    def from_source(
        cls,
        source: str,
        bm_index_path: str,
        qdrant_index_path: Optional[str],
        collection_name: Optional[str],
        embedding_model: Optional[str],
        bm25_candidates: Optional[int],
        dense_candidates: Optional[int],
        topk: Optional[int],
        expected_embedding_dim: Optional[int],
        qdrant_host: Optional[str],
        qdrant_port: Optional[int],
        qdrant_api_key: Optional[str],
    ) -> "HybridSearchSettings":
        defaults = _get_source_defaults(source)
        return cls(
            source=source,
            bm_index_path=Path(bm_index_path),
            qdrant_index_path=Path(qdrant_index_path) if qdrant_index_path else None,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            qdrant_api_key=qdrant_api_key,
            collection_name=collection_name or defaults["collection"],
            embedding_model=embedding_model or defaults["embedding_model"],
            embedding_dim=expected_embedding_dim if expected_embedding_dim is not None else defaults["embedding_dim"],
            bm25_candidates=bm25_candidates or defaults["bm25_candidates"],
            dense_candidates=dense_candidates or defaults["dense_candidates"],
            topk=topk or defaults["topk"],
        )


class HybridSearchService:
    """Stateful helper that keeps handles to the BM25 index, Qdrant client, and embedding model."""

    def __init__(self, settings: HybridSearchSettings):
        self.settings = settings
        self._bm25_index = open_bm25_index(str(settings.bm_index_path))
        self._qdrant = connect_qdrant(
            qdrant_index_path=str(settings.qdrant_index_path) if settings.qdrant_index_path else None,
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
        )
        self._validate_embedding_setup()

    @classmethod
    @lru_cache(maxsize=4)
    def get_cached(cls, settings: HybridSearchSettings) -> "HybridSearchService":
        return cls(settings)

    @property
    def model(self) -> SentenceTransformer:
        return load_embed_model(self.settings.embedding_model)

    @property
    def qdrant(self) -> QdrantClient:
        return self._qdrant

    def search(
        self,
        query: str,
        *,
        bm25_candidates: Optional[int] = None,
        dense_candidates: Optional[int] = None,
        topk: Optional[int] = None,
        return_payloads: bool = True,
    ) -> Dict:
        bm_limit = bm25_candidates or self.settings.bm25_candidates
        dense_limit = dense_candidates or self.settings.dense_candidates
        top_limit = topk or self.settings.topk

        bm_hits = bm25_search(self._bm25_index, query, topk=bm_limit)
        dense_hits = dense_search(
            self._qdrant,
            self.model,
            query,
            self.settings.collection_name,
            topk=dense_limit,
        )

        fused = rrf_fuse(bm_hits, dense_hits, k=60)
        top_ids = [cid for cid, _ in fused[:top_limit]]

        bm_scores = {cid: score for cid, score in bm_hits}
        dense_scores = {cid: score for cid, score in dense_hits}

        payloads = (
            fetch_payloads_for_ids(self._qdrant, self.settings.collection_name, top_ids)
            if return_payloads
            else {}
        )

        results = []
        fused_scores = dict(fused)
        for cid in top_ids:
            item = {
                "chunk_id": cid,
                "fused_score": fused_scores.get(cid, 0.0),
                "bm25_score": bm_scores.get(cid, 0.0),
                "dense_score": dense_scores.get(cid, 0.0),
            }
            if return_payloads:
                payload = payloads.get(cid)
                if payload:
                    text = (payload.get("text") or "").replace("\n", " ")
                    item.update(
                        {
                            "paper_id": payload.get("paper_id"),
                            "section": payload.get("section"),
                            "source_file": payload.get("source_file"),
                            "start_line": payload.get("start_line"),
                            "end_line": payload.get("end_line"),
                            "labels": payload.get("labels"),
                            "equations_raw": payload.get("equations_raw"),
                            "text": text,
                        }
                    )
            results.append(item)

        return {
            "query": query,
            "results": results,
            "diagnostics": {
                "bm25_candidates": len(bm_hits),
                "dense_candidates": len(dense_hits),
                "bm_index_path": str(self.settings.bm_index_path.resolve()),
                "qdrant_mode": "embedded"
                if self.settings.qdrant_index_path
                else f"server:{self.settings.qdrant_host}:{self.settings.qdrant_port or 6333}",
                "collection_name": self.settings.collection_name,
                "embedding_model": self.settings.embedding_model,
                "source": self.settings.source,
            },
        }

    def _validate_embedding_setup(self) -> None:
        model_dim = self.model.get_sentence_embedding_dimension()
        _ensure_embedding_dim_matches(
            self._qdrant,
            self.settings.collection_name,
            self.settings.embedding_model,
            model_dim,
            self.settings.embedding_dim,
        )


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
        points, _ = qdrant.scroll(collection_name=collection_name, scroll_filter=flt, limit=1)
        if points:
            out[cid] = points[0].payload
    return out


# ---------- TOP-LEVEL HYBRID SEARCH ----------


def hybrid_search_from_disk(
    query: str,
    bm_index_path: str,
    qdrant_index_path: Optional[str] = None,
    qdrant_host: Optional[str] = None,
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
    settings = HybridSearchSettings.from_source(
        source=source,
        bm_index_path=bm_index_path,
        qdrant_index_path=qdrant_index_path,
        collection_name=collection_name,
        embedding_model=embedding_model,
        bm25_candidates=bm25_candidates,
        dense_candidates=dense_candidates,
        topk=topk,
        expected_embedding_dim=expected_embedding_dim,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        qdrant_api_key=qdrant_api_key,
    )
    service = HybridSearchService.get_cached(settings)
    return service.search(
        query,
        bm25_candidates=bm25_candidates,
        dense_candidates=dense_candidates,
        topk=topk,
        return_payloads=return_payloads,
    )


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

    raise ValueError(f"Unable to determine vector size for collection '{collection_name}'.")


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


def hybrid_txt_search_from_disk(*args, **kwargs):
    """Backward compatible alias for callers expecting the TXT-specific function name."""
    kwargs.setdefault("source", "txt")
    return hybrid_search_from_disk(*args, **kwargs)


# ---------- Final Rerank ----------


def rerank(query, items, text_key: str = "text", topk: int = 10, model_name: str = "BAAI/bge-reranker-large"):
    """items = [{'chunk_id':..., 'text':..., ...}, ...]"""
    reranker = _load_reranker(model_name)
    pairs = [(query, it.get(text_key, "")) for it in items]
    scores = reranker.predict(pairs)  # higher is better
    ranked = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)
    return [it for it, _ in ranked[:topk]]
