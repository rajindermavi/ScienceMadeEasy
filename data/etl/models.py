from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PaperMeta(BaseModel):
    """Metadata we track for each arXiv paper in the pipeline."""

    arxiv_id: str
    base_id: str
    sanitized_id: str
    version: str
    title: str
    primary_category: str
    categories: List[str]
    authors: List[str]
    published_date: str
    updated_date: str
    url: str
    summary: Optional[str] = None
    comment: Optional[str] = None
    citation_list: Optional[List] = None
    reference_list: Optional[List] = None  

class Chunk(BaseModel):
    """Smaller text unit carved out of the TEI body for downstream indexing."""

    chunk_id: str
    arxiv_id: str
    version: str
    order: int = Field(..., ge=0)
    start_paragraph: int = Field(..., ge=0)
    end_paragraph: int = Field(..., ge=0)
    token_count: int = Field(..., ge=0)
    text: str


class ChunkDiagnostics(BaseModel):
    """Lightweight diagnostics returned alongside the generated chunks."""

    arxiv_id: str
    version: str
    source_path: str
    total_paragraphs: int = 0
    total_chunks: int = 0
    discarded_paragraphs: int = 0
