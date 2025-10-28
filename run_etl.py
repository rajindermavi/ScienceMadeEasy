import logging
import os
import json
from pydantic import BaseModel
from pathlib import Path

from data.etl.etl_stage_a import build_arxiv_query, arxiv_extract
from data.etl.etl_stage_b import prepare_latex_corpus, latex_conversion
from data.etl.etl_stage_c_md import md_collection_chunking
from data.etl.etl_stage_c_txt import txt_collection_chunking
from data.etl.etl_stage_d_md import index_md_bm25, index_md_qdrant
from data.etl.etl_stage_d_txt import index_txt_bm25, index_txt_qdrant
import config

from logs.logger import get_logger
logger = get_logger(log_name='run_etl',log_path=config.DEFAULT_LOG_DIR/'etl.log')

def json_default(o):
    if isinstance(o, BaseModel):
        return o.model_dump()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

def run_arxiv_to_latex_extract(phrases, categories, max_results):
    logger.info(
        "Starting run_arxiv_extract | phrases=%s | categories=%s | max_results=%s",
        phrases,
        categories,
        max_results,
    )

    out = {}

    logger.info("Calling build_arxiv_query")
    arxiv_query = build_arxiv_query(phrases, categories)
    out["arviv_query"] = arxiv_query
    logger.info("build_arxiv_query completed")

    logger.info("Calling arxiv_extract")
    papers = arxiv_extract(arxiv_query, max_results=max_results)
    # out["papers"] = papers
    logger.info("arxiv_extract conpleted")

    logger.info("Calling prepare_latex_corpus")
    combined_latex_paths = prepare_latex_corpus({arxiv_id: paper.get('latex_dir') for arxiv_id, paper in papers.items()})
    for arxiv_id, paper in papers.items():
        paper['combined_latex_path'] = combined_latex_paths[arxiv_id]
    logger.info("prepare_latex_corpus produced")

    logger.info("Calling latex_conversion")
    conversions = latex_conversion(combined_latex_paths)
    for arxiv_id, paper in papers.items():
        paper['md_full_path'] = conversions.get(arxiv_id,{}).get('md')
        paper['txt_full_path'] = conversions.get(arxiv_id,{}).get('txt')

    out['papers'] = papers

    return out

def run_chunking(papers):
    logger.info("Starting run_chunking")

    out = {}
    md_filepaths = {arxiv_id:paper.get('md_full_path') for arxiv_id, paper in papers.items()}
    logger.info("Calling md_collection_chunking")
    md_chunking=md_collection_chunking(md_filepaths) # fix to dict
    out["md_details"] = md_chunking['md_details']
    logger.info("md_collection_chunking complete")

    txt_filepaths = {arxiv_id:paper.get('txt_full_path') for arxiv_id, paper in papers.items()}
    logger.info("Calling txt_collection_chunking")
    txt_chunking = txt_collection_chunking(txt_filepaths) # fix to dict
    out["txt_details"] = txt_chunking['txt_details']
    logger.info("txt_collection_chunking complete")

    for arxiv_id, paper in papers.items():
        paper['md_chunk_files'] = md_chunking.get('md_chunk_files',{}).get(arxiv_id)
        paper['txt_chunk_files'] = txt_chunking.get('txt_chunk_files',{}).get(arxiv_id)

    logger.info("run_arxiv_extract complete")

    return out


def run_indexing():
    logger.info("Starting run_indexing")

    out = {}

    md_jsonl = config.MD_JSONL
    md_bm25_index_dir = config.MD_BM25_INDEX_DIR
    md_qdrant_index_dir = config.MD_QDRANT_INDEX_DIR
    txt_jsonl = config.TXT_JSONL
    txt_bm25_index_dir = config.TXT_BM25_INDEX_DIR
    txt_qdrant_index_dir = config.TXT_QDRANT_INDEX_DIR
    

    logger.info("Calling index_md_bm25")
    out["md_bm25"] = index_md_bm25(md_jsonl, md_bm25_index_dir)

    logger.info("Calling index_md_qdrant")
    out["md_qdrant"] = index_md_qdrant(md_jsonl, md_qdrant_index_dir)

    logger.info("Calling index_txt_bm25")
    out["txt_bm25"] = index_txt_bm25(txt_jsonl, txt_bm25_index_dir)

    logger.info("Calling index_txt_qdrant")
    out["txt_bm25"] = index_txt_qdrant(txt_jsonl, txt_qdrant_index_dir)

    logger.info(
        "Indexing complete | bm25=%s | qdrant=%s",
        out["md_bm25"],
        out["md_qdrant"],
    )

    logger.info("run_indexing complete")

    return out


if __name__ == "__main__":
    logger.info("ETL process started")

    phrases = [
        "almost Mathieu operator",
        "Aubry-André",
        "Aubry André",
        "Harper model",
        "quasiperiodic Schrodinger operators",
        "ergodic Schrodinger operators",
    ]

    categories = ["math-ph", "math.SP", "quant-ph"]

    arxiv_extract_details = run_arxiv_to_latex_extract(phrases, categories, 200)
    papers = arxiv_extract_details.get('papers')
    
    chunking_details = run_chunking(papers)
    arxiv_extract_details.update(chunking_details)

    indexing_details = run_indexing()

    arxiv_extract_details.update(indexing_details)
    report_string = json.dumps(
        arxiv_extract_details, 
        indent=4,
        default=json_default
    )

    with open(config.EXTRACT_DETAILS,"w") as f:
        json.dump(
            arxiv_extract_details,
            f,
            indent = 4,
            default=json_default
        )
    report_heading = '\n'*5 + 'EXTRACT REPORT' + '\n'*5
    logger.info(f'{report_heading} {report_string}')

    logger.info("ETL process completed")
