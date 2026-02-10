# app.py — Minimal RAG Frontend (Streamlit)
# ---------------------------------------------------------------

from dotenv import load_dotenv
load_dotenv()
import os

import os
import streamlit as st

from openai import OpenAI

from query.rag_agent import (
    AgentState, 
    agent,
    QARecord,
    SessionMemory,
    ChunkBelief,
    apply_feedback
)
from query.nlp import render_response

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI()

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="SME RAG", layout="wide")
st.title("Science Made Easy")

# ---------------------------
# Session state
# ---------------------------
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if "current_query" not in st.session_state:
    st.session_state.current_query = ""

# ---------------------------
# Render previous (frozen) Q&A
# ---------------------------
for idx, item in enumerate(st.session_state.qa_history):
    st.text_area(
        label="Previous query",
        value=item["query"],
        height=120,
        disabled=True,
        key=f"frozen_query_{idx}",
        label_visibility="collapsed",
    )
    st.button(
        "Submitted",
        use_container_width=True,
        disabled=True,
        key=f"frozen_submit_{idx}",
    )
    answer_string = item.get("answer", "")
    render_response(answer_string)
    citations = item.get("citations","")
    citation_lines = [f"[{key}]: {reference}" for key, reference in citations.items()]
    st.markdown("<br>".join(citation_lines), unsafe_allow_html=True)

st.divider()

# ---------------------------
# Current (active) Q&A input
# ---------------------------
with st.form("query_form", clear_on_submit=False):
    query = st.text_area(
        "Enter your question/query",
        height=120,
        placeholder="e.g., Describe the spectral properties of the Almost Mathieu Operator",
        key="current_query",
    )
    do_search = st.form_submit_button(
        "Submit Query",
        use_container_width=True,
        disabled=st.session_state.get("is_submitting", False),
    )

if do_search and query.strip():
    st.session_state.is_submitting = True
    initial_state = AgentState(
        query=query,
        k=10,
        max_k=20,
    )

    result = agent.invoke(initial_state)
    answer = result.get("answer", "")
    citations = result.get("citations", "")

    st.session_state.qa_history.append(
        {
            "query": query,
            "answer": answer,
            "citations": citations
        }
    )

    st.session_state.is_submitting = False
    st.rerun()