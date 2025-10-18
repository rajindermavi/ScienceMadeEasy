import gzip
import shutil
import tarfile
import time
import re
from pathlib import Path
from typing import Iterable, List, Set
from urllib.error import HTTPError

import requests
import arxiv

import config
from config import ARXIV_EPRINT, ARXIV_ID_RE, ARXIV_PDF, GZIP_MAGIC, RAW_DIR
from data.etl.models import PaperMeta

## ####################
## Prepare Arxiv Search
## ####################

def build_arxiv_query(phrases: Iterable[str], categories: Iterable[str]) -> str:
    """
    Build an arXiv advanced query that:
      - searches for any of the phrases across all metadata (title/abstract/etc.)
      - restricts results to one or more subject categories
    """
    phrase_part = " OR ".join(f'all:"{p}"' for p in phrases)
    cat_part = " OR ".join(f"cat:{c}" for c in categories)
    return f"({phrase_part}) AND ({cat_part})"

## ###########################
## Search Papers and Meta Data
## ###########################

def arxiv_client_search(page_size, delay_seconds, query, max_results, sort_by, sort_order):
    client = arxiv.Client(
        page_size=page_size,
        delay_seconds=delay_seconds,
        num_retries=3,
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    return list(client.results(search))


def parse_arxiv_ids(entry_id: str) -> tuple[str, str, str]:
    """Extract base identifier, version, and a filesystem-safe filename stem."""
    m = ARXIV_ID_RE.search(entry_id)
    if m:
        base_id = m.group(1)
        version = f"v{m.group(2)}" if m.group(2) else "v1"
    else:
        tail = entry_id.rsplit("/", 1)[-1]
        if "v" in tail:
            base_id, ver = tail.split("v", 1)
            version = f"v{ver}"
        else:
            base_id, version = tail, "v1"

    numeric_part = base_id.split("/")[-1] or base_id
    sanitized = f"{numeric_part}{version}"
    return base_id, version, sanitized

def get_semantic_scholar_data(arxiv_id: str) -> int:

    url_cite = config.URL_SEMANTIC_SCHOLAR_CIT.format(arxiv_id=arxiv_id)
    url_ref = config.URL_SEMANTIC_SCHOLAR_REF.format(arxiv_id=arxiv_id)

    params = {"fields": "title,year,venue,externalIds,authors,url"}
    headers = {"User-Agent": "refs-fetch/1.0"}

    resp_cite = requests.get(url_cite, params=params, headers=headers, timeout=20)
    resp_ref = requests.get(url_ref, params=params, headers=headers, timeout=20)

    if resp_cite.ok:
        cite = resp_cite.json().get('data',[])
    else:
        cite = []

    if resp_ref.ok:
        ref = resp_ref.json().get('data',[])
    else:
        ref = []

    return cite, ref

def semantic_scholar_arxiv_ids(semantic_scholar_collection):

    arxiv_ids = []
    for paper in semantic_scholar_collection:
        arxiv_ids.append(paper.get('externalIds',{}).get('ArXiv'))
    
    return arxiv_ids

## ###########
## COORDINATOR
## ###########

def arxiv_metas(
    arxiv_query: str,
    max_results: int = 200,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
    sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
    page_size: int = 100,
    delay_seconds: float = 0.5,
) -> List[PaperMeta]:
    arxiv_search_results = arxiv_client_search(page_size, delay_seconds, arxiv_query, max_results, sort_by, sort_order)

    seen: Set[str] = set()
    metas: List[PaperMeta] = []
    
    for arxiv_search_result in arxiv_search_results:
        base_id, version, sanitized_id = parse_arxiv_ids(arxiv_search_result.entry_id)
        arxiv_id = arxiv_search_result.get_short_id()
        if arxiv_id in seen:
            continue
        seen.add(arxiv_id)

        citation_list, reference_list = get_semantic_scholar_data(base_id) 

        updated_date = arxiv_search_result.updated.isoformat() if arxiv_search_result.updated else arxiv_search_result.published.isoformat()
        metas.append(
            PaperMeta(
                arxiv_id=arxiv_id,
                base_id=base_id,
                sanitized_id=sanitized_id,
                version=version,
                title=arxiv_search_result.title.strip(),
                primary_category=arxiv_search_result.primary_category,
                categories=list(arxiv_search_result.categories),
                authors=[a.name for a in arxiv_search_result.authors],
                published_date=arxiv_search_result.published.isoformat(),
                updated_date=updated_date,
                url=arxiv_search_result.entry_id,
                summary=arxiv_search_result.summary,
                comment=arxiv_search_result.comment,
                citation_list=semantic_scholar_arxiv_ids(citation_list),
                reference_list=semantic_scholar_arxiv_ids(reference_list)
            )
        )
        time.sleep(0.5)
    return metas

## ###############
## DOWNLOAD PAPERS
## ###############

def save_stream(resp: requests.Response, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 15):
            if chunk:
                fh.write(chunk)

def is_gzip_file(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(len(GZIP_MAGIC)) == GZIP_MAGIC
    except OSError:
        return False

def download_latex(base_id: str, sanitized_id: str, version: str) -> Path | None:
    """Fetch the LaTeX source archive for an arXiv paper and unpack it."""

    tar_path = Path(config.TAR_DIR) / f"{sanitized_id}.tar"
    tar_path.parent.mkdir(parents=True, exist_ok=True)

    extract_dir = Path(config.TAR_EXTRACT_DIR) / sanitized_id
    extract_dir.mkdir(parents=True, exist_ok=True)

    header_name = _ensure_source_archive(base_id, version, tar_path)
    if header_name is None:
        return None

    archive_root = extract_dir.resolve()
    if tarfile.is_tarfile(tar_path):
        _extract_tar_archive(tar_path, archive_root)
    elif is_gzip_file(tar_path):
        _extract_single_gzip(tar_path, archive_root, header_name, sanitized_id)
    else:
        _write_plain_tex(tar_path, archive_root, header_name, sanitized_id)

    return extract_dir

def _ensure_source_archive(base_id: str, version: str, tar_path: Path) -> str | None:
    """Download the source archive if needed, returning the response header."""

    if tar_path.exists() and tar_path.stat().st_size > 0:
        return ""

    if tar_path.exists():
        tar_path.unlink()

    candidate_ids = [f"{base_id}{version}"]
    if "/" in base_id:
        candidate_ids.append(base_id)

    for candidate in candidate_ids:
        url = ARXIV_EPRINT.format(id=candidate)
        response = requests.get(url, stream=True, allow_redirects=True, timeout=30)
        if response.status_code == 200:
            header = response.headers.get("content-disposition", "")
            save_stream(response, tar_path)
            return header or ""

    print(f"error on {base_id}{version}")
    return None


def _safe_resolved_path(base: Path, member_name: str) -> Path | None:
    """Resolve `member_name` under `base`, guarding against path traversal."""

    target = (base / member_name).resolve()
    if not target.is_relative_to(base):
        return None
    return target


def _extract_tar_archive(tar_path: Path, extract_root: Path) -> None:
    with tarfile.open(tar_path, mode="r:*") as tf:
        for member in tf.getmembers():
            target = _safe_resolved_path(extract_root, member.name)
            if target is None:
                continue
            tf.extract(member, path=extract_root)


def _infer_inner_name(header_name: str, sanitized_id: str) -> str:
    match = re.search(r'filename="?([^";]+)"?', header_name or "")
    inner_name = match.group(1) if match and match.group(1) else f"{sanitized_id}.tex"
    if inner_name.endswith(".gz"):
        inner_name = inner_name[:-3]
    if not inner_name.lower().endswith(".tex"):
        inner_name = f"{inner_name}.tex"
    return Path(inner_name).name


def _extract_single_gzip(tar_path: Path, extract_root: Path, header_name: str, sanitized_id: str) -> None:
    target_name = _infer_inner_name(header_name, sanitized_id)
    target = _safe_resolved_path(extract_root, target_name)
    if target is None:
        return

    with gzip.open(tar_path, "rb") as gz, open(target, "wb") as out:
        shutil.copyfileobj(gz, out)


def _write_plain_tex(tar_path: Path, extract_root: Path, header_name: str, sanitized_id: str) -> None:
    raw_bytes = tar_path.read_bytes()
    if b"\x00" in raw_bytes:
        print(f"[warn] {tar_path} is neither a tar archive nor plain text")
        return

    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = raw_bytes.decode("latin-1")

    target_name = _infer_inner_name(header_name, sanitized_id)
    target = _safe_resolved_path(extract_root, target_name)
    if target is None:
        print(f"[warn] refusing to write outside extract dir for {tar_path}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(raw_text, encoding="utf-8")

## ################
## COMPLETE EXTRACT
## ################

def arxiv_extract(arxiv_query, max_results):
    
    paper_metas=arxiv_metas(arxiv_query, max_results)
    papers = {}

    for meta in paper_metas:

        latex_dir = download_latex(meta.base_id,meta.sanitized_id,meta.version)
        if latex_dir is None:
            print(f"[warn] unable to fetch source for {meta.base_id}")
            continue
        papers[meta.arxiv_id] = {
            'meta':meta,
            'latex_dir':latex_dir 
        }
    
    return papers
