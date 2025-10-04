import json
from tqdm import tqdm

from data.etl.etl_stage_a import build_arxiv_query, arxiv_extract
from data.etl.etl_stage_b import prepare_latex_corpus, latex_conversion
from data.etl.etl_stage_c_md import md_collection_chunking
from data.etl.etl_stage_c_txt import txt_collection_chunking

def run_arxiv_extract(phrases, categories, max_results):

    out = {}

    arxiv_query = build_arxiv_query(phrases,categories)
    out['arviv_query'] = arxiv_query

    paper_metas = arxiv_extract(arxiv_query,max_results = max_results)
    out['paper_metas'] = paper_metas

    written_paths = prepare_latex_corpus()
    out['written_paths'] = written_paths

    md_paths, txt_paths = latex_conversion(written_paths)
    out['md_paths'] = md_paths
    out['txt_paths'] = txt_paths

    md_chunked_files = md_collection_chunking(md_paths)
    out['md_chunked_files'] = md_chunked_files

    txt_chunked_files = txt_collection_chunking(txt_paths)
    out['txt_chunked_files'] = txt_chunked_files

    return out

if __name__ == "__main__":
    phrases = [
        "almost Mathieu operator",
        "Aubry-André",           # hyphen form
        "Aubry André",           # space form (some metadata lacks the hyphen)
        "Harper model",           # common synonym
        "quasiperiodic Schrodinger operators"
    ]

    categories = ["math-ph", "math.SP", "quant-ph"] 

    out = run_arxiv_extract(phrases,categories,25)