import os
from dotenv import load_dotenv

from openai import OpenAI

from query.index_query import hybrid_search_from_disk, rerank
from query.rag import is_mathy, build_context_blocks, format_context_md, build_prompt
import config

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI()

def query_context(query,token_budget = 1800):
# Embedded Qdrant (folder-based)
    resp_md = hybrid_search_from_disk(
        query=query,
        bm_index_path=config.MD_BM25_INDEX_DIR,
        qdrant_index_path=config.MD_QDRANT_INDEX_DIR,
        collection_name=config.MD_QDRANT_COLLECTION,
        embedding_model=config.MD_EMBEDDING_MODEL,
        topk=config.MD_TOPK,
        source="md",
        return_payloads=True
    )
    resp_txt = hybrid_search_from_disk(
        query=query,
        bm_index_path=config.TXT_BM25_INDEX_DIR,
        qdrant_index_path=config.TXT_QDRANT_INDEX_DIR,
        collection_name=config.TXT_QDRANT_COLLECTION,
        embedding_model=config.TXT_EMBEDDING_MODEL,
        topk=config.TXT_TOPK,
        source="txt",
        return_payloads=True
    )

    results = [*resp_md['results'],*resp_txt['results']]

    rerank_results = rerank(query,results)
    blocks = build_context_blocks(rerank_results, max_tokens=token_budget)
    context_str = format_context_md(blocks)

    return context_str

def llm(query,context_str):

    query = query or 'Explain spectral statistics of the almost Mathiu operator for Liouville frequencies'
    math_mode = is_mathy(query)

    prompt = build_prompt(query, context_str, math_mode)

    response = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
    )

    return response.output_text

def rag(query):
    context_str = query_context(query)
    response = llm(query,context_str)
    return response

if __name__ == "__main__":
    query = 'Explain spectral statistics for Liouville frequencies'
    response = rag(query)
    print(response)
