import os
import re
from pathlib import Path

DATA_ETL_DIR = Path(os.getenv("DATA_ETL_DIR","data/data_etl"))
DATA_ETL_DIR.mkdir(parents=True,exist_ok=True)
DATA_INDEX_DIR = Path(os.getenv("DATA_INDEX_DIR","data/data_index"))
DATA_INDEX_DIR.mkdir(parents=True,exist_ok=True)
RAW_DIR = DATA_INDEX_DIR / "pdf_raw"
RAW_DIR.mkdir(parents=True,exist_ok=True)
TAR_DIR = DATA_INDEX_DIR / "tar"
TAR_DIR.mkdir(parents=True,exist_ok=True)
TAR_EXTRACT_DIR = DATA_INDEX_DIR / "latex_raw"
TAR_EXTRACT_DIR.mkdir(parents=True,exist_ok=True)
LATEX_FILTER_DIR = DATA_INDEX_DIR / "latex_final"
LATEX_FILTER_DIR.mkdir(parents=True,exist_ok=True)
MD_VERSION_DIR = DATA_INDEX_DIR / "full_markdown"
MD_VERSION_DIR.mkdir(parents=True,exist_ok=True)
MD_CHUNKED_DIR = DATA_INDEX_DIR / "md_chunked"
MD_CHUNKED_DIR.mkdir(parents=True,exist_ok=True)
TEXT_VERSION_DIR = DATA_INDEX_DIR / "full_text"
TEXT_VERSION_DIR.mkdir(parents=True,exist_ok=True)
TXT_CHUNKED_DIR = DATA_INDEX_DIR / "txt_chunked"
TXT_CHUNKED_DIR.mkdir(parents=True,exist_ok=True)

MD_JSONL = DATA_INDEX_DIR / "md_data.jsonl"
TXT_JSONL = DATA_INDEX_DIR / "txt_data.jsonl"

MD_BM25_INDEX_DIR = DATA_INDEX_DIR / 'md_bm25_storage'
MD_BM25_INDEX_DIR.mkdir(parents=True,exist_ok=True)
MD_QDRANT_INDEX_DIR = DATA_INDEX_DIR / 'md_qrant_storage'
MD_QDRANT_INDEX_DIR.mkdir(parents=True,exist_ok=True)

ARXIV_PDF = "https://arxiv.org/pdf/{id}.pdf"          # accepts old and new IDs; version optional
ARXIV_EPRINT = "https://arxiv.org/e-print/{id}"       # source tarball fallback (last resort)
ARXIV_ID_RE = re.compile(r"arxiv\.org\/abs\/([0-9]+\.[0-9]+|[a-z\-]+\/[0-9]+)v?(\d+)?")
GZIP_MAGIC = b"\x1f\x8b"