# app.py — Minimal RAG Frontend (Streamlit)
# ---------------------------------------------------------------

from dotenv import load_dotenv
load_dotenv()
import os

import uuid

import streamlit as st
import json
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

from log.logger import get_logger
logger = get_logger(log_name= 'app', log_path = 'app.log')

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

# ---------------------------------
# Get topic summary
# --------------------------------

from query.rag_agent import retriever
from query.prompt import topic_summary_system_prompt, llm_topic_summary
from query.llm import LLM
topics = retriever.topics
summary_prompt = llm_topic_summary(topics)
summary = LLM.generate_text(topic_summary_system_prompt,summary_prompt)

st.subheader(f'This RAG is prepared to answer questions on the following topic(s): {topics}.')

render_response(summary)

# ---------------------------
# Render previous (frozen) Q&A
# ---------------------------
for idx, qa_rec in enumerate(st.session_state.qa_history):
    st.text_area(
        label="Previous query",
        value=qa_rec.query,
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
    answer_string = qa_rec.answer
    render_response(answer_string)
    citations = qa_rec.citations
    citation_lines = [f"[{key}]: {reference}" for key, reference in citations.items()]
    st.markdown("<br>".join(citation_lines), unsafe_allow_html=True)
    feedback_labels = ["Downvote", "Neutral", "Upvote"]
    feedback_map = {"Downvote": -1, "Neutral": 0, "Upvote": 1}
    current_label = "Neutral"
    if qa_rec.user_feedback in (-1, 0, 1):
        current_label = feedback_labels[qa_rec.user_feedback + 1]
    selected_label = st.radio(
        "Feedback",
        feedback_labels,
        index=feedback_labels.index(current_label),
        horizontal=True,
        key=f"feedback_{idx}",
        label_visibility="collapsed",
    )
    qa_rec.user_feedback = feedback_map[selected_label]


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

    qa_record = QARecord(
        question_id=str(uuid.uuid4()),
        query=query,
        used_chunk_ids=result.get("used_chunk_ids", []),
        provenance=result.get("provenance", {}),
        judgment=result.get("sufficient", True),
        answer=result.get("answer", "Answer Missing"),
        citations=result.get("citations", {}),
    )
    logger.info(
        json.dumps(
            result,
            indent=2,
            default=lambda o: list(o) if isinstance(o, set) else str(o),
        )
    )


    st.session_state.qa_history.append(qa_record)

    st.session_state.is_submitting = False
    st.rerun()

