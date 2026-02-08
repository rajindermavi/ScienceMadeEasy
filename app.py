# app.py — Minimal RAG Frontend (Streamlit)
# ---------------------------------------------------------------
# Features:
# - Text box for query
# - Uses your stored BM25 (Whoosh) + Qdrant (embedded/server) indexes
# - Calls your existing `hybrid_search_from_disk` function
# - Shows fused results with full text and metadata
# - Builds a RAG prompt you can copy/paste into your LLM client
# - Optional live LLM call if OPENAI_API_KEY is set (commented template)
# ---------------------------------------------------------------

import os
import re
import textwrap
import streamlit as st
from dotenv import load_dotenv

from openai import OpenAI

#from query.index_query import hybrid_search_from_disk, rerank
#from query.rag_utils import is_mathy, build_context_blocks, format_context_md, build_prompt

from query.rag_agent import (
    AgentState, 
    agent,
    QARecord,
    SessionMemory,
    ChunkBelief,
    apply_feedback
)
# 
# from etl.config import (
#     MD_TOPK, 
#     TXT_TOPK, 
#     MD_BM25_INDEX_DIR,
#     MD_QDRANT_INDEX_DIR,
#     MD_QDRANT_COLLECTION,
#     MD_EMBEDDING_MODEL,
#     TXT_BM25_INDEX_DIR,
#     TXT_QDRANT_INDEX_DIR,
#     TXT_QDRANT_COLLECTION,
#     TXT_EMBEDDING_MODEL
# )
# 
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI()

# ---------------------------
# Helpers
# ---------------------------
#def query_context(query,token_budget = 1800):
## Embedded Qdrant (folder-based)
#    resp_md = hybrid_search_from_disk(
#        query=query,
#        bm_index_path=MD_BM25_INDEX_DIR,
#        qdrant_index_path=MD_QDRANT_INDEX_DIR,
#        collection_name=MD_QDRANT_COLLECTION,
#        embedding_model=MD_EMBEDDING_MODEL,
#        topk=MD_TOPK,
#        source="md",
#        return_payloads=True
#    )
#    resp_txt = hybrid_search_from_disk(
#        query=query,
#        bm_index_path=TXT_BM25_INDEX_DIR,
#        qdrant_index_path=TXT_QDRANT_INDEX_DIR,
#        collection_name=TXT_QDRANT_COLLECTION,
#        embedding_model=TXT_EMBEDDING_MODEL,
#        topk=TXT_TOPK,
#        source="txt",
#        return_payloads=True
#    )
#
#    results = [*resp_md['results'],*resp_txt['results']]
#
#    rerank_results = rerank(query,results)
#    blocks = build_context_blocks(rerank_results, max_tokens=token_budget)
#    context_str = format_context_md(blocks)
#
#    return context_str
#
#def llm(query,context_str):
#
#    query = query or 'Explain spectral statistics for quasiperiodic schrodinger operators with diophantine frequencies in the critical regime'
#    math_mode = is_mathy(query)
#
#    prompt = build_prompt(query, context_str, math_mode)
#
#    response = client.responses.create(
#        model="gpt-4o",
#        input=[
#            {"role": "system", "content": prompt["system"]},
#            {"role": "user", "content": prompt["user"]},
#        ],
#    )
#
#    return response.output_text
#
#
#def render_response(raw_text: str) -> None:
#    """Render plain Markdown with LaTeX segments handled via st.latex."""
#    normalized = raw_text.replace("\\\\", "\\")
#    normalized = normalized.replace(r"\(", "$").replace(r"\)", "$")
#    tokens = re.split(r"(\\\[.*?\\\]|\\\(.*?\\\))", normalized, flags=re.DOTALL)
#    for token in tokens:
#        if not token:
#            continue
#        if token.startswith("\\[") and token.endswith("\\]"):
#            st.latex(token[2:-2].strip())
#        elif token.startswith("\\(") and token.endswith("\\)"):
#            st.latex(token[2:-1].strip())
#        else:
#            st.markdown(token)


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="SME RAG", layout="wide")
st.title("Science Made Easy")

# Query box
query = st.text_area("Enter your question/query", height=120, placeholder="e.g., Explain spectral statistics for quasiperiodic schrodinger operators with diophantine frequencies in the critical regime")

# Search button
do_search = st.button("Submit Query", use_container_width=True)

if do_search:

    #context_str = query_context(query)

    #response = llm(query,context_str)

    initial_state = AgentState(
        query=query,
        k=10,
        max_k = 20
    )

    result = agent.invoke(initial_state)
    st.subheader("Results")
    st.markdown(result.get("answer", ""))
    st.markdown(result.get("citationsr", ""))
    #render_response(result)
