import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


#def _resolve_path(env_var: str, default_relative: str) -> Path:
#    """Resolve a path to an absolute location under the project root."""
#    raw_value = os.getenv(env_var, default_relative)
#    path = Path(raw_value).expanduser()
#    if not path.is_absolute():
#        path = PROJECT_ROOT / path
#    return path.resolve()


DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"  #_resolve_path("DEFAULT_LOG_DIR", "logs")
DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)


DATA_ETL_DIR = PROJECT_ROOT / 'data/data_etl' # _resolve_path("DATA_ETL_DIR", "data/data_etl")
DATA_ETL_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DETAILS = DATA_ETL_DIR / 'extract_details.json'
DATA_INDEX_DIR = PROJECT_ROOT / "data/data_index" # _resolve_path("DATA_INDEX_DIR", "data/data_index")
DATA_INDEX_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = DATA_ETL_DIR / "pdf_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
TAR_DIR = DATA_ETL_DIR / "tar"
TAR_DIR.mkdir(parents=True, exist_ok=True)
TAR_EXTRACT_DIR = DATA_ETL_DIR / "latex_raw"
TAR_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
LATEX_FILTER_DIR = DATA_ETL_DIR / "latex_final"
LATEX_FILTER_DIR.mkdir(parents=True, exist_ok=True)
MD_VERSION_DIR = DATA_ETL_DIR / "full_markdown"
MD_VERSION_DIR.mkdir(parents=True, exist_ok=True)
MD_CHUNKED_DIR = DATA_ETL_DIR / "md_chunked"
MD_CHUNKED_DIR.mkdir(parents=True, exist_ok=True)
TEXT_VERSION_DIR = DATA_ETL_DIR / "full_text"
TEXT_VERSION_DIR.mkdir(parents=True, exist_ok=True)
TXT_CHUNKED_DIR = DATA_ETL_DIR / "txt_chunked"
TXT_CHUNKED_DIR.mkdir(parents=True, exist_ok=True)

MD_JSONL = DATA_ETL_DIR / "md_data.jsonl"
TXT_JSONL = DATA_ETL_DIR / "txt_data.jsonl"

MD_BM25_INDEX_DIR = DATA_INDEX_DIR / 'md_bm25_storage'
MD_BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
MD_QDRANT_INDEX_DIR = DATA_INDEX_DIR / 'md_qrant_storage'
MD_QDRANT_INDEX_DIR.mkdir(parents=True, exist_ok=True)
TXT_BM25_INDEX_DIR = DATA_INDEX_DIR / 'txt_bm25_storage'
TXT_BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
TXT_QDRANT_INDEX_DIR = DATA_INDEX_DIR / 'txt_qrant_storage'
TXT_QDRANT_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# MD Indexing
MD_QDRANT_COLLECTION = os.getenv("MD_QDRANT_COLLECTION","md_chunks")
MD_EMBEDDING_MODEL = os.getenv("MD_EMBEDDING_MODEL","BAAI/bge-small-en")
MD_EMBEDDING_DIM = int(os.getenv("MD_EMBEDDING_DIM","384"))
MD_QDRANT_BATCH_SIZE = int(os.getenv("MD_QDRANT_BATCH_SIZE","64"))
MD_QDRANT_PROGRESS_INTERVAL = int(os.getenv("MD_QDRANT_PROGRESS_INTERVAL","64"))
MD_BM25_CANDIDATES = int(os.getenv("MD_BM25_CANDIDATES","200"))
MD_DENSE_CANDIDATES = int(os.getenv("MD_DENSE_CANDIDATES","200"))
MD_TOPK = int(os.getenv("MD_TOPK","10"))

# TXT Indexing
TXT_QDRANT_COLLECTION = os.getenv("TXT_QDRANT_COLLECTION","txt_chunks")
TXT_EMBEDDING_MODEL = os.getenv("TXT_EMBEDDING_MODEL","BAAI/bge-base-en-v1.5")
TXT_EMBEDDING_DIM = int(os.getenv("TXT_EMBEDDING_DIM","768"))
TXT_QDRANT_BATCH_SIZE = int(os.getenv("TXT_QDRANT_BATCH_SIZE","64"))
TXT_QDRANT_PROGRESS_INTERVAL = int(os.getenv("TXT_QDRANT_PROGRESS_INTERVAL","64"))
TXT_BM25_CANDIDATES = int(os.getenv("TXT_BM25_CANDIDATES","200"))
TXT_DENSE_CANDIDATES = int(os.getenv("TXT_DENSE_CANDIDATES","200"))
TXT_TOPK = int(os.getenv("TXT_TOPK","10"))

ARXIV_PDF = "https://arxiv.org/pdf/{id}.pdf"          # accepts old and new IDs; version optional
ARXIV_EPRINT = "https://arxiv.org/e-print/{id}"       # source tarball fallback (last resort)
ARXIV_ID_RE = re.compile(r"arxiv\.org\/abs\/([0-9]+\.[0-9]+|[a-z\-]+\/[0-9]+)v?(\d+)?")
GZIP_MAGIC = b"\x1f\x8b"

# URL
URL_SEMANTIC_SCHOLAR_REF = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}/references"
URL_SEMANTIC_SCHOLAR_CIT = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}/citations"
