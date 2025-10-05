import os
import re
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR","data/data"))
DATA_DIR.mkdir(parents=True,exist_ok=True)
RAW_DIR = DATA_DIR / "pdf_raw"
RAW_DIR.mkdir(parents=True,exist_ok=True)
TAR_DIR = DATA_DIR / "tar"
TAR_DIR.mkdir(parents=True,exist_ok=True)
TAR_EXTRACT_DIR = DATA_DIR / "latex_raw"
TAR_EXTRACT_DIR.mkdir(parents=True,exist_ok=True)
LATEX_FILTER_DIR = DATA_DIR / "latex_final"
LATEX_FILTER_DIR.mkdir(parents=True,exist_ok=True)
MD_VERSION_DIR = DATA_DIR / "full_markdown"
MD_VERSION_DIR.mkdir(parents=True,exist_ok=True)
MD_CHUNKED_DIR = DATA_DIR / "md_chunked"
MD_CHUNKED_DIR.mkdir(parents=True,exist_ok=True)
TEXT_VERSION_DIR = DATA_DIR / "full_text"
TEXT_VERSION_DIR.mkdir(parents=True,exist_ok=True)
TXT_CHUNKED_DIR = DATA_DIR / "txt_chunked"
TXT_CHUNKED_DIR.mkdir(parents=True,exist_ok=True)

MD_JSONL = DATA_DIR / "md_data.jsonl"
TXT_JSONL = DATA_DIR / "txt_data.jsonl"

ARXIV_PDF = "https://arxiv.org/pdf/{id}.pdf"          # accepts old and new IDs; version optional
ARXIV_EPRINT = "https://arxiv.org/e-print/{id}"       # source tarball fallback (last resort)
ARXIV_ID_RE = re.compile(r"arxiv\.org\/abs\/([0-9]+\.[0-9]+|[a-z\-]+\/[0-9]+)v?(\d+)?")
GZIP_MAGIC = b"\x1f\x8b"