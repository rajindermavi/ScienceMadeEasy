import logging
import os

from data.etl.etl_stage_a import build_arxiv_query, arxiv_extract
from data.etl.etl_stage_b import prepare_latex_corpus, latex_conversion
from data.etl.etl_stage_c_md import md_collection_chunking
from data.etl.etl_stage_c_txt import txt_collection_chunking
from data.etl.etl_stage_d_md import index_md_bm25, index_md_qdrant
from data.etl.etl_stage_d_txt import index_txt_bm25, index_txt_qdrant
import config


def get_etl_logger():
    """Configure and return the ETL logger."""
    logger = logging.getLogger("etl")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    log_dir = os.path.join(os.path.dirname(__file__), "logging")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "etl.log")

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def run_arxiv_extract(phrases, categories, max_results):
    logger = get_etl_logger()
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
    paper_metas = arxiv_extract(arxiv_query, max_results=max_results)
    out["paper_metas"] = paper_metas
    logger.info("arxiv_extract returned %s results", len(paper_metas))

    logger.info("Calling prepare_latex_corpus")
    written_paths = prepare_latex_corpus()
    out["written_paths"] = written_paths
    logger.info("prepare_latex_corpus produced %s paths", len(written_paths))

    logger.info("Calling latex_conversion")
    md_paths, txt_paths = latex_conversion(written_paths)
    out["md_paths"] = md_paths
    out["txt_paths"] = txt_paths
    logger.info(
        "latex_conversion produced %s markdown paths and %s text paths",
        len(md_paths),
        len(txt_paths),
    )

    logger.info("Calling md_collection_chunking")
    md_details= md_collection_chunking(md_paths)
    out["md_details"] = md_details
    md_chunked_files = md_details['chunk_files_written']
    logger.info("md_collection_chunking created %s files", md_chunked_files)

    logger.info("Calling txt_collection_chunking")
    txt_details = txt_collection_chunking(txt_paths)
    out["txt_details"] = txt_details
    txt_chunked_files = txt_details['chunk_files_written']
    logger.info("txt_collection_chunking created %s files", len(txt_chunked_files))

    logger.info("run_arxiv_extract complete")

    return out


def run_indexing():
    logger = get_etl_logger()
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
    logger = get_etl_logger()
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

    out_etl = run_arxiv_extract(phrases, categories, 500)
    out_indexing = run_indexing()

    report = {**out_etl, **out_indexing}

    logger.info(report)

    logger.info("ETL process completed")
