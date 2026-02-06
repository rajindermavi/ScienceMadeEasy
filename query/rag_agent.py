
from typing import List, Set, Optional
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

from query.retrieval import IndexRetrieval
from query.rag_utils import llm_answer
from query.nlp import get_top_scoring_segments_as_string
from query.prompt import llm_judge_sufficiency_prompt, llm_judge_sufficiency_response_schema
from query.llm import LLM 

# SESSION MANAGEMENT

class QARecord(BaseModel):
    question_id: str
    query: str
    used_chunks: list[str]
    provenance: dict[str, str]  # chunk_id → "search" | "neighbor"
    judgment: bool              # agent thought sufficient
    answer: str
    user_feedback: Optional[int] = None  # e.g. -1, 0, +1

class ChunkBelief(BaseModel):
    support_count: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    last_used_in: list[str] = []  # question_ids

class SessionMemory(BaseModel):
    chunk_stats: dict[str, ChunkBelief] = {}
    question_links: dict[str, set[str]] = {}  # question_id → chunks

def apply_feedback(record: QARecord, session: SessionMemory):
    if record.user_feedback is None:
        return

    for cid in record.used_chunks:
        belief = session.chunk_stats.setdefault(cid, ChunkBelief())
        if record.user_feedback > 0:
            belief.positive_feedback += 1
        elif record.user_feedback < 0:
            belief.negative_feedback += 1


# SINGLE QUERY AGENT

retriever = IndexRetrieval()

## ------------------ RETRIEVAL ------------------ ##
class AgentState(BaseModel):
    # user intent
    query: str

    # retrieval control
    k: int = 10
    max_k: int = 20
    search_round: int = 0
    max_search_rounds: int = 3

    # retrieved data
    retrieved_chunks: List[str] = Field(default_factory=list)
    frontier_chunks: List[str] = Field(default_factory=list)
    visited_chunks: Set[str] = Field(default_factory=set)

    # judgments
    sufficient: Optional[bool] = None
    sufficiency_reason: Optional[str] = None

    # memory (epistemic, session-scoped)
    remembered_chunks: Set[str] = Field(default_factory=set)
    rejected_chunks: Set[str] = Field(default_factory=set)

    answer: str = ""

    # control flags
    stop: bool = False

def search_index(state: AgentState) -> AgentState:
    chunks = retriever.search(state.query, k=state.k)

    new_ids = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id:
                new_ids.append(chunk_id)
    return {
        "retrieved_chunks": new_ids,
        "frontier_chunks": new_ids,
        "visited_chunks": state.visited_chunks | set(new_ids)
    }

## ------------------ LLM JUDGMENT ------------------ ##

class SufficiencyVerdict(BaseModel):
    sufficient: bool
    reason: str

def llm_judge(query: str, chunk_ids: List[str]) -> SufficiencyVerdict:
    representative_texts = []
    for chunk_id in chunk_ids:
        chunk = retriever.chunks.get(chunk_id)
        text = chunk.get('text', '') if chunk else ''
        top_segments = get_top_scoring_segments_as_string(text)
        representative_texts.append(top_segments)
    prompt = llm_judge_sufficiency_prompt(query, representative_texts)

    response = LLM.generate_json(prompt, llm_judge_sufficiency_response_schema)

    verdict = SufficiencyVerdict(**response)
    return verdict

def judge_sufficiency(state: AgentState) -> AgentState:
    verdict = llm_judge(
        query=state.query,
        chunk_ids=state.retrieved_chunks
    )
    return {
        "sufficient": verdict.sufficient,
        "sufficiency_reason": verdict.reason
    }

## ------------------ CONTROL LOGIC ------------------ #

def decide_next_step(state: AgentState) -> AgentState:
    if state.sufficient:
        return {"stop": True}

    if state.search_round >= state.max_search_rounds:
        return {"stop": True}

    if state.k < state.max_k:
        return {
            "k": min(state.k + 5, state.max_k),
            "search_round": state.search_round + 1
        }

    return {"stop": True}

## ------------------ ANSWER SYNTHESIS ------------------ #

def synthesize_answer(state: AgentState):

    meta = {}

    for chunk_id in state.retrieved_chunks:
        chunk_data = {}
        chunk = retriever.chunks.get(chunk_id)
        paper_id = chunk.get('paper_id')
        chunk_data['chunk_id'] = chunk_id        
        chunk_data['paper_id'] = paper_id 
        chunk_data['text'] = chunk.get('text')
        chunk_data['eqns'] = chunk.get('equations_raw',None)
        paper = retriever.papers.get(paper_id)
        paper_meta = paper.get('meta')
        chunk_data['title'] = paper_meta.get('title')
        chunk_data['url'] = paper_meta.get('url')
        meta[chunk_id] = chunk_data

    answer = llm_answer(
        state.query,
        chunk_data
    )

    return {
        "answer": answer
    }

## ------------------ AGENT CONSTRUCTION ------------------ ##

agent_builder = StateGraph(AgentState)

agent_builder.add_node("search", search_index)
agent_builder.add_node("judge", judge_sufficiency)
agent_builder.add_node("decide", decide_next_step)
agent_builder.add_node("answer", synthesize_answer)

agent_builder.set_entry_point("search")

agent_builder.add_edge("search", "judge")
agent_builder.add_edge("judge", "decide")

agent_builder.add_conditional_edges(
    "decide",
    lambda state: "answer" if state.stop else "search"
)

agent_builder.add_edge("answer", END)

agent = agent_builder.compile()
