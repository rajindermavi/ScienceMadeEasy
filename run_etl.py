import json
import sys
from pydantic import BaseModel
from pathlib import Path

from log.logger import  get_logger
logger = get_logger(log_path=f'{Path(__file__).stem}.log', level='INFO')

DEFAULT_PHRASES = [
    "almost Mathieu operator",
    "Aubry-André",
    "Aubry André",
    "Harper model",
    "quasiperiodic Schrodinger operators",
    "ergodic Schrodinger operators",
]
DEFAULT_CATEGORIES = ["math-ph", "math.SP", "quant-ph"]
DEFAULT_MAX_PAPERS = 300

def json_default(o):
    if isinstance(o, BaseModel):
        return o.model_dump()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

def run_arxiv_extract(phrases, categories, max_results):
    logger.info(
        "Starting %s | phrases=%s | categories=%s | max_results=%s",
        sys._getframe().f_code.co_name,
        phrases,
        categories,
        max_results,
    )
    from etl.stg_a_extract import build_arxiv_query, arxiv_extract


    out = {}

    logger.info("Calling build_arxiv_query")
    arxiv_query = build_arxiv_query(phrases, categories)
    out["arviv_query"] = arxiv_query
    logger.info("build_arxiv_query completed")

    logger.info("Calling arxiv_extract")
    papers = arxiv_extract(arxiv_query, max_results=max_results)
    logger.info("arxiv_extract conpleted")
    total_papers = len(papers)
    logger.info(f'Collected {total_papers} papers.')

    out['papers'] = papers

    return out

def run_transform_latex(papers):
    logger.info(f'Run {sys._getframe().f_code.co_name}.')
    from etl.stg_b_conversion import prepare_latex_corpus, latex_conversion

    out = {}

    logger.info("Calling prepare_latex_corpus")
    combined_latex_paths = prepare_latex_corpus({arxiv_id: paper.get('latex_dir') for arxiv_id, paper in papers.items()})
    logger.info("Calling latex_conversion")
    conversions = latex_conversion(combined_latex_paths)

    for arxiv_id, paper in papers.items():
        paper['combined_latex_path'] = combined_latex_paths.get(arxiv_id)
        paper['md_full_path'] = conversions.get(arxiv_id,{}).get('md_file')
        paper['txt_full_path'] = conversions.get(arxiv_id,{}).get('txt_file')    

    out['papers'] = papers

    logger.info("run_transform_latex complete")

    return out

def run_chunking(papers):
    logger.info(f'Run {sys._getframe().f_code.co_name}.')
    from etl.stg_c_md_chunking import md_collection_chunking
    from etl.stg_c_txt_chunking import txt_collection_chunking

    md_filepaths = {arxiv_id:paper.get('md_full_path') for arxiv_id, paper in papers.items()}
    logger.info("Calling md_collection_chunking")
    md_chunking=md_collection_chunking(md_filepaths) # fix to dict
    logger.info("md_collection_chunking complete")

    txt_filepaths = {arxiv_id:paper.get('txt_full_path') for arxiv_id, paper in papers.items()}
    logger.info("Calling txt_collection_chunking")
    txt_chunking = txt_collection_chunking(txt_filepaths) # fix to dict
    logger.info("txt_collection_chunking complete")

    for arxiv_id, paper in papers.items():
        paper['md_recs'] = md_chunking.get('normed_records',{}).get(arxiv_id)
        paper['txt_recs'] = txt_chunking.get('normed_records',{}).get(arxiv_id)

    logger.info("run_arxiv_extract complete")



def run_indexing():
    logger.info(f'Run {sys._getframe().f_code.co_name}.')
    from etl.stg_d_md_indexing import index_md_bm25, index_md_qdrant
    from etl.stg_d_txt_indexing import index_txt_bm25, index_txt_qdrant

    out = {}
    from etl.config import (
        MD_JSON,
        MD_BM25_INDEX_DIR,
        TXT_JSON,
        TXT_BM25_INDEX_DIR,
    )
    md_json =  MD_JSON
    md_bm25_index_dir =  MD_BM25_INDEX_DIR
    txt_json =  TXT_JSON
    txt_bm25_index_dir =  TXT_BM25_INDEX_DIR

    logger.info("Calling index_md_bm25")
    out["md_bm25"] = index_md_bm25(md_json, md_bm25_index_dir)

    logger.info("Calling index_md_qdrant")
    out["md_qdrant"] = index_md_qdrant(md_json)

    logger.info("Calling index_txt_bm25")
    out["txt_bm25"] = index_txt_bm25(txt_json, txt_bm25_index_dir)

    logger.info("Calling index_txt_qdrant")
    out["txt_qdrant"] = index_txt_qdrant(txt_json)

    logger.info('\tmd_bm25: %s',out["md_bm25"])
    logger.info('\tmd_qdrant: %s',out["md_qdrant"])
    logger.info('\ttxt_bm25: %s',out["txt_bm25"])
    logger.info('\ttxt_qdrant: %s',out["txt_qdrant"])

    logger.info("run_indexing complete")

    return out

def run_etl(stages='abcd'):
    logger.info("ETL process started")

    if 'a' in stages:
        logger.info("Stage A: Arxiv Extraction")
        phrases,categories,max_papers = DEFAULT_PHRASES,DEFAULT_CATEGORIES,DEFAULT_MAX_PAPERS

        arxiv_extract_details = run_arxiv_extract(phrases, categories, max_papers)
        papers = arxiv_extract_details.get('papers')  
        #arxiv_extract_details['paper_index'] = list(papers.keys())
        from etl.config import STAGE_A
        with open( STAGE_A,"w") as f:
            json.dump(
                arxiv_extract_details,
                f,
                indent = 4,
                default=json_default
            )

    if 'b' in stages:  
        from etl.config import STAGE_A, STAGE_B
        with open( STAGE_A, 'r') as file:
            arxiv_extract_details = json.load(file)
        papers = arxiv_extract_details.get('papers')  
        transform_details = run_transform_latex(papers) 
        arxiv_extract_details.update(transform_details)

        with open( STAGE_B,"w") as f:
            json.dump(
                arxiv_extract_details,
                f,
                indent = 4,
                default=json_default
            )

    if 'c' in stages:   
        from etl.config import STAGE_B, STAGE_C
        with open( STAGE_B, 'r') as file:
            arxiv_extract_details = json.load(file)

        papers = arxiv_extract_details.get('papers')

        run_chunking(papers)

        with open( STAGE_C,"w") as f:
            json.dump(
                arxiv_extract_details,
                f,
                indent = 4,
                default=json_default
            )

    if 'd' in stages:
        from etl.config import STAGE_C, EXTRACT_DETAILS
        with open( STAGE_C, 'r') as file:
            arxiv_extract_details = json.load(file)

        indexing_details = run_indexing()

        arxiv_extract_details.update(indexing_details)

        # with open( STAGE_D,"w") as f:
        #     json.dump(
        #         arxiv_extract_details,
        #         f,
        #         indent = 4,
        #         default=json_default
        #     )

        with open( EXTRACT_DETAILS,"w") as f:
            json.dump(
                arxiv_extract_details,
                f,
                indent = 4,
                default=json_default
            )

    report_string = json.dumps(
        arxiv_extract_details, 
        indent=4,
        default=json_default
    )

    report_heading = '\n'*5 + 'EXTRACT REPORT' + '\n'*5
    logger.info(f'{report_heading} {report_string}')

    logger.info("ETL process completed")



if __name__ == "__main__":
    run_etl()
