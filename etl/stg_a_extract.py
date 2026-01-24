import gzip
import shutil
import tarfile
import time
import sys
from typing import Iterable, List, Set

import requests
import arxiv

import etl.config as config
from etl.config import (
    ARXIV_ID_RE,
    URL_SEMANTIC_SCHOLAR_CIT,
    URL_SEMANTIC_SCHOLAR_REF,
    TAR_DIR,
    TAR_EXTRACT_DIR,
    GZIP_MAGIC,
)
from etl.models import PaperMeta

from pathlib import Path
from log.logger import get_logger
logger = get_logger(log_path=Path(__file__).stem, level='INFO')

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

def arxiv_client_search(query, max_results):
    logger.info(f'Run {sys._getframe().f_code.co_name}.')
    page_size = 25
    delay_seconds = 1
    batch_size = 50
    total_collected = 0
    collected = []
    max_iterations = max_results//batch_size + 1
    # print(max_iterations)
    def _iterate_results(results):
        collected = []

        # for r in results: 
        while True:  
            try:
                r = next(results)
                collected.append(r)
            except StopIteration:
                logger.info(f'End of batch. Collected: {len(collected)}. StopIteration.')
                break
            except Exception as e:
                logger.info(f'End of batch. Collected: {len(collected)}.')
                
                logger.info(f'\t\tException: {e}.')
                #print(f'End of batch. Exception {e}. Collected: {len(collected)}.')
                break  
        return collected

    client = arxiv.Client(
        page_size=page_size,
        delay_seconds=delay_seconds,
        num_retries=3,
    )

    for _ in range(max_iterations):
        search = arxiv.Search(
                query=query,
                max_results=batch_size+total_collected,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending,
        )
        results = client.results(search,offset=total_collected)
        batch = _iterate_results(results)
        #print(f'batch size: {len(batch)}')
        if len(batch) == 0:
            break
        total_collected += len(batch)
        collected.extend(batch)
        time.sleep(3)
    logger.info(f'Completed {sys._getframe().f_code.co_name}.')
    return collected

def parse_arxiv_ids(entry_id: str) -> tuple[str, str, str]:
    logger.debug(f'Run {sys._getframe().f_code.co_name}. input {entry_id}')
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
    logger.debug(f'Run {sys._getframe().f_code.co_name}. input {arxiv_id}')
    url_cite = URL_SEMANTIC_SCHOLAR_CIT.format(arxiv_id=arxiv_id)
    url_ref = URL_SEMANTIC_SCHOLAR_REF.format(arxiv_id=arxiv_id)

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
    """
        Get references and citations from Semantic Scholar collection
    """
    # logger.debug(f'Run {sys._getframe().f_code.co_name}. input {}')
    arxiv_ids = []
    for paper in semantic_scholar_collection or []:
        id = None
        if 'citingPaper' in paper:
            paper_ngbr = paper.get('citingPaper',{}).get('externalIds',{})
        if 'citedPaper' in paper:
            paper_ngbr = paper.get('citedPaper',{}).get('externalIds',{})
        if paper_ngbr:
            id = paper_ngbr.get('ArXiv')
        if not id == None and not id in arxiv_ids: 
            arxiv_ids.append(id)
    
    return arxiv_ids

## ###########
## COORDINATOR
## ###########

def arxiv_metas(
    arxiv_query: str,
    max_results: int = 300
) -> List[PaperMeta]:
    logger.info(f'Run {sys._getframe().f_code.co_name}.')
    arxiv_search_results = arxiv_client_search(arxiv_query, max_results)

    seen: Set[str] = set()
    metas: List[PaperMeta] = []
    logger.info(f'In {sys._getframe().f_code.co_name}. Download Sources')
    idx = 0 
    dl_idx = 0
    for arxiv_search_result in arxiv_search_results:
        
        base_id, version, sanitized_id = parse_arxiv_ids(arxiv_search_result.entry_id)
        arxiv_id = arxiv_search_result.get_short_id().replace('/','_')
        if arxiv_id in seen:
            continue
        seen.add(arxiv_id)
        try:
            gzip_filename = f"{arxiv_id}.tar.gz"
            arxiv_search_result.download_source(
                dirpath=str(TAR_DIR), 
                filename=gzip_filename
            )
            # logger.info(f'Download success: arxiv {arxiv_id}.')
            extract_success = True
            dl_idx += 1
        except:
            gzip_filename = ''
            logger.info(f'Download failed: arxiv {arxiv_id}.')
            extract_success = False
            pass

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
                gzip=gzip_filename,
                summary=arxiv_search_result.summary,
                comment=arxiv_search_result.comment,
                extract=extract_success,
                citation_list=semantic_scholar_arxiv_ids(citation_list),
                reference_list=semantic_scholar_arxiv_ids(reference_list)
            )
        )
        time.sleep(0.5)
        idx += 1
        if idx % 25 == 0:
            logger.info(f'Download progress. Attempted download of {idx} records. Downloaded {dl_idx} successfully.')
    logger.info(f'Download End. Attempted download of {idx} records. Downloaded {dl_idx} successfully.')
    logger.info(f'Complete {sys._getframe().f_code.co_name}.')
    return metas

## ###############
## DOWNLOAD PAPERS
## ###############

def is_gzip_file(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(len(GZIP_MAGIC)) == config.GZIP_MAGIC
    except OSError:
        return False


def extract_tarfile(filename,paper_id):
    file_path = TAR_DIR / filename
    if not file_path.exists():
        logger.info(f'** Download missing. File {filename}.')
        return None    
    extract_root = TAR_EXTRACT_DIR / paper_id
    extract_root.mkdir(parents=True, exist_ok=True)
    target_root = extract_root.resolve()

    def _is_within_root(path: Path) -> bool:
        try:
            path.relative_to(target_root)
            return True
        except ValueError:
            return False

    if tarfile.is_tarfile(file_path):
        try:
            with tarfile.open(file_path, mode="r:*") as tf:
                for member in tf.getmembers():
                    member_path = (extract_root / member.name).resolve()
                    if not _is_within_root(member_path):
                        logger.info(f'** Extract tarfile skipped. File {filename}. Member {member.name} outside target root.')
                        continue
                    if member.isdir():
                        member_path.mkdir(parents=True, exist_ok=True)
                        continue
                    if member.issym() or member.islnk():
                        logger.info(f'** Extract tarfile skipped link. File {filename}. Member {member.name}.')
                        continue
                    member_path.parent.mkdir(parents=True, exist_ok=True)
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        continue
                    try:
                        with open(member_path, "wb") as out_fh:
                            shutil.copyfileobj(extracted, out_fh)
                    finally:
                        extracted.close()
            
        except Exception as e:
            logger.info(f'** Extract tarfile. File {filename}. Exception {e}.')
            extract_root = None
    elif is_gzip_file(file_path):
        try:
            extract_tex = extract_root / f'{paper_id}.tex'
            
            with gzip.open(file_path, "rb") as gz, open(extract_tex, "wb") as out:
                shutil.copyfileobj(gz, out)
        except Exception as e:
            logger.info(f'** Extract gzip. File {filename}. Exception {e}.')
            extract_root = None
    else:
        logger.info(f'Extract file neither tarfile or gzip. File {filename}..')
        extract_root = None
    return extract_root

# def save_stream(resp: requests.Response, out_path: Path):
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(out_path, "wb") as fh:
#         for chunk in resp.iter_content(chunk_size=1 << 15):
#             if chunk:
#                 fh.write(chunk)
# 
# def is_gzip_file(path: Path) -> bool:
#     try:
#         with open(path, "rb") as fh:
#             return fh.read(len(GZIP_MAGIC)) == GZIP_MAGIC
#     except OSError:
#         return False
# 
# def download_latex(base_id: str, sanitized_id: str, version: str) -> Path | None:
#     """Fetch the LaTeX source archive for an arXiv paper and unpack it."""
# 
#     tar_path = Path(config.TAR_DIR) / f"{sanitized_id}.tar"
#     tar_path.parent.mkdir(parents=True, exist_ok=True)
# 
#     extract_dir = Path(config.TAR_EXTRACT_DIR) / sanitized_id
#     extract_dir.mkdir(parents=True, exist_ok=True)
# 
#     header_name = _ensure_source_archive(base_id, version, tar_path)
#     if header_name is None:
#         return None
# 
#     archive_root = extract_dir.resolve()
#     if tarfile.is_tarfile(tar_path):
#         _extract_tar_archive(tar_path, archive_root)
#     elif is_gzip_file(tar_path):
#         _extract_single_gzip(tar_path, archive_root, header_name, sanitized_id)
#     else:
#         _write_plain_tex(tar_path, archive_root, header_name, sanitized_id)
# 
#     return extract_dir
# 
# def _ensure_source_archive(base_id: str, version: str, tar_path: Path) -> str | None:
#     """Download the source archive"""
# 
# 
#     candidate_ids = [f"{base_id}{version}"]
#     if "/" in base_id:
#         candidate_ids.append(base_id)
# 
#     for candidate in candidate_ids:
#         url = ARXIV_EPRINT.format(id=candidate)
#         response = requests.get(url, stream=True, allow_redirects=True, timeout=30)
#         if response.status_code == 200:
#             header = response.headers.get("content-disposition", "")
#             save_stream(response, tar_path)
#             return header or ""
# 
#     logger.info(f"error on {base_id}{version}")
#     return None
# 
# 
# def _safe_resolved_path(base: Path, member_name: str) -> Path | None:
#     """Resolve `member_name` under `base`, guarding against path traversal."""
# 
#     target = (base / member_name).resolve()
#     if not target.is_relative_to(base):
#         return None
#     return target
# 
# 
# def _extract_tar_archive(tar_path: Path, extract_root: Path) -> None:
#     with tarfile.open(tar_path, mode="r:*") as tf:
#         for member in tf.getmembers():
#             target = _safe_resolved_path(extract_root, member.name)
#             if target is None:
#                 continue
#             tf.extract(member, path=extract_root)
# 
# 
# def _infer_inner_name(header_name: str, sanitized_id: str) -> str:
#     match = re.search(r'filename="?([^";]+)"?', header_name or "")
#     inner_name = match.group(1) if match and match.group(1) else f"{sanitized_id}.tex"
#     if inner_name.endswith(".gz"):
#         inner_name = inner_name[:-3]
#     if not inner_name.lower().endswith(".tex"):
#         inner_name = f"{inner_name}.tex"
#     return Path(inner_name).name
# 
# 
# def _extract_single_gzip(tar_path: Path, extract_root: Path, header_name: str, sanitized_id: str) -> None:
#     target_name = _infer_inner_name(header_name, sanitized_id)
#     target = _safe_resolved_path(extract_root, target_name)
#     if target is None:
#         return
# 
#     with gzip.open(tar_path, "rb") as gz, open(target, "wb") as out:
#         shutil.copyfileobj(gz, out)
# 
# 
# def _write_plain_tex(tar_path: Path, extract_root: Path, header_name: str, sanitized_id: str) -> None:
#     raw_bytes = tar_path.read_bytes()
#     if b"\x00" in raw_bytes:
#         logger.info(f"[warn] {tar_path} is neither a tar archive nor plain text")
#         return
# 
#     try:
#         raw_text = raw_bytes.decode("utf-8")
#     except UnicodeDecodeError:
#         raw_text = raw_bytes.decode("latin-1")
# 
#     target_name = _infer_inner_name(header_name, sanitized_id)
#     target = _safe_resolved_path(extract_root, target_name)
#     if target is None:
#         logger.info(f"[warn] refusing to write outside extract dir for {tar_path}")
#         return
# 
#     target.parent.mkdir(parents=True, exist_ok=True)
#     target.write_text(raw_text, encoding="utf-8")

## ################
## COMPLETE EXTRACT
## ################

def arxiv_extract(arxiv_query, max_results):
    logger.info(f'Run {sys._getframe().f_code.co_name}.')
    
    paper_metas=arxiv_metas(arxiv_query, max_results)
    logger.info(f'Total metas: {len(paper_metas)}.')

    papers = {}
    logger.info('Extract Tarfiles ... }.')
    for meta in paper_metas:

        if meta.extract:
            latex_dir = extract_tarfile(meta.gzip,meta.arxiv_id)
            papers[meta.arxiv_id] = {
                 'meta':meta,
                 'extract':True,
                 'latex_dir':latex_dir 
            }
            logger.info(f'Extract Success. arxiv {meta.arxiv_id}. latex_dir {latex_dir}.')
        else:
            papers[meta.arxiv_id] = {
                 'meta':meta,
                 'extract':False,
                 'latex_dir':None
            }
            logger.info(f'Extract Failed. arxiv {meta.arxiv_id}.')

    return papers
