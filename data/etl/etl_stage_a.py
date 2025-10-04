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




def is_gzip_file(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(len(GZIP_MAGIC)) == GZIP_MAGIC
    except OSError:
        return False


def build_arxiv_query(phrases: Iterable[str], categories: Iterable[str]) -> str:
    """
    Build an arXiv advanced query that:
      - searches for any of the phrases across all metadata (title/abstract/etc.)
      - restricts results to one or more subject categories
    """
    phrase_part = " OR ".join(f'all:"{p}"' for p in phrases)
    cat_part = " OR ".join(f"cat:{c}" for c in categories)
    return f"({phrase_part}) AND ({cat_part})"


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


def save_stream(resp: requests.Response, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 15):
            if chunk:
                fh.write(chunk)


def download_pdf(result, out_path: Path, sleep: float = 0.2) -> Path | None:
    """
    Try arxiv.Result.download_pdf first; on HTTP 404 fall back to unversioned URL
    and finally to e-print (source). Returns path or None.
    """
    out_path = Path(out_path)
    if out_path.exists():
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_id, version, _ = parse_arxiv_ids(result.entry_id)

    # 1) Try the library helper (uses versioned pdf_url).
    try:
        result.download_pdf(filename=str(out_path))
        time.sleep(sleep)
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
    except HTTPError as exc:
        if getattr(exc, "code", None) != 404:
            raise

    # 2) Fallback: unversioned PDF URL (serves latest version)
    unversioned = ARXIV_PDF.format(id=base_id)
    resp = requests.get(unversioned, stream=True, allow_redirects=True, timeout=30)
    if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("application/pdf"):
        save_stream(resp, out_path)
        return out_path

    return None


def download_latex(base_id: str, sanitized_id: str, version: str) -> Path | None:
    tar_path = Path(config.TAR_DIR) / f"{sanitized_id}.tar"
    tar_path.parent.mkdir(parents=True, exist_ok=True)

    header_name = ""
    if tar_path.exists():
        pass
    else:
        candidate_ids = [f"{base_id}{version}"]
        if "/" in base_id:
            candidate_ids.append(base_id)

        for candidate in candidate_ids:
            response = requests.get(ARXIV_EPRINT.format(id=candidate), stream=True, allow_redirects=True, timeout=30)
            if response.status_code == 200:
                header_name = response.headers.get("content-disposition", "")
                save_stream(response, tar_path)
                break
        else:
            print(f"error on {base_id}{version}")
            return None

    extract_dir = Path(config.TAR_EXTRACT_DIR) / sanitized_id
    extract_dir.mkdir(parents=True, exist_ok=True)

    if tarfile.is_tarfile(tar_path):

        with tarfile.open(tar_path, mode="r:*") as tf:
            for member in tf.getmembers():
                target = (extract_dir / member.name).resolve()
                if not target.is_relative_to(extract_dir.resolve()):
                    continue
                tf.extract(member, path=extract_dir)

    elif is_gzip_file(tar_path):
        with gzip.open(tar_path, "rb") as gz:
            match = re.search(r'filename="?([^";]+)"?', header_name)
            if match and match.group(1):
                inner_name = match.group(1)
            else:
                inner_name = f"{sanitized_id}.tex"
            if inner_name.endswith(".gz"):
                inner_name = inner_name[:-3]
            if not inner_name.lower().endswith(".tex"):
                inner_name = f"{inner_name}.tex"
            target = (extract_dir / Path(inner_name).name).resolve()
            if target.is_relative_to(extract_dir.resolve()):
                with open(target, "wb") as out:
                    shutil.copyfileobj(gz, out)
    else:
        # Some legacy e-prints ship a single TeX file (no archive compression).
        raw_bytes = tar_path.read_bytes()
        if b"\x00" in raw_bytes:
            print(f"[warn] {tar_path} is neither a tar archive nor plain text")
            return None
        try:
            raw_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raw_text = raw_bytes.decode("latin-1")

        match = re.search(r'filename="?([^";]+)"?', header_name)
        inner_name = match.group(1) if match and match.group(1) else f"{sanitized_id}.tex"
        if not inner_name.lower().endswith(".tex"):
            inner_name = f"{inner_name}.tex"
        inner_path = (extract_dir / Path(inner_name).name).resolve()
        if not inner_path.is_relative_to(extract_dir.resolve()):
            print(f"[warn] refusing to write outside extract dir for {tar_path}")
            return None
        inner_path.parent.mkdir(parents=True, exist_ok=True)
        inner_path.write_text(raw_text, encoding="utf-8")

    return extract_dir


def get_citations(arxiv_id: str) -> int:
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=citationCount"
    resp = requests.get(url)
    if resp.ok:
        return resp.json().get("citationCount", 0)
    return 0


def arxiv_extract(
    arxiv_query: str,
    max_results: int = 200,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
    sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
    page_size: int = 100,
    delay_seconds: float = 0.5,
) -> List[PaperMeta]:
    arxiv_search_results = arxiv_client_search(page_size, delay_seconds, arxiv_query, max_results, sort_by, sort_order)

    seen: Set[str] = set()
    out: List[PaperMeta] = []

    for arxiv_search_result in arxiv_search_results:
        base_id, version, sanitized_id = parse_arxiv_ids(arxiv_search_result.entry_id)
        arxiv_id = arxiv_search_result.get_short_id()
        if arxiv_id in seen:
            continue
        seen.add(arxiv_id)

        latex_dir = download_latex(base_id, sanitized_id, version)
        if latex_dir is None:
            print(f"[warn] unable to fetch source for {arxiv_search_result.entry_id}")
            continue

        pdf_path = Path(config.RAW_DIR) / f"{sanitized_id}.pdf"
        pdf_path.mkdir(parents=True,exist_ok=True)
        pdf_file = download_pdf(arxiv_search_result, pdf_path)
        if pdf_file is None:
            print(f"[warn] unable to fetch PDF for {arxiv_search_result.entry_id}")
            continue

        citations = get_citations(base_id)
        out.append(
            PaperMeta(
                arxiv_id=arxiv_id,
                base_id=base_id,
                version=version,
                title=arxiv_search_result.title.strip(),
                primary_category=arxiv_search_result.primary_category,
                categories=list(arxiv_search_result.categories),
                authors=[a.name for a in arxiv_search_result.authors],
                published_date=arxiv_search_result.published.isoformat(),
                updated_date=arxiv_search_result.updated.isoformat() if arxiv_search_result.updated else arxiv_search_result.published.isoformat(),
                url=arxiv_search_result.entry_id,
                summary=arxiv_search_result.summary,
                comment=arxiv_search_result.comment,
                citations=citations,
                pdf_path=str(pdf_file),
                latex_dir=str(latex_dir),
            )
        )
        time.sleep(0.5)

    return out
