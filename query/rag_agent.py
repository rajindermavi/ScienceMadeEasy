
from typing import List, Set, Optional, Dict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

from query.retrieval import IndexRetrieval
from query.nlp import get_top_scoring_segments_as_string
from query.prompt import (
    llm_judge_sufficiency_system_prompt,
    llm_judge_sufficiency_prompt, 
    llm_judge_sufficiency_response_schema,
    final_answer_system_prompt,
    final_answer_user_prompt,
    final_answer_response_schema
)
from query.llm import LLM 

from log.logger import get_logger
logger = get_logger(log_name= 'rag_agent', log_path = 'rag_agent.log')

# SESSION MANAGEMENT

class QARecord(BaseModel):
    question_id: str
    query: str
    used_chunk_ids: list[str]
    provenance: dict[str, str]  # chunk_id → "search" | "neighbor"
    judgment: bool              # agent thought sufficient
    answer: str
    citations: dict[str,str]
    user_feedback: Optional[int] = None  # e.g. -1, 0, +1

class ChunkBelief(BaseModel):
    support_count: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    last_used_in: list[str] = []  # question_ids

class SessionMemory(BaseModel):
    chunk_stats: dict[str, ChunkBelief] = Field(default_factory=dict)
    questions: dict[str, str] = Field(default_factory=dict)  # question_id → question
    question_links: dict[str, set[str]] = Field(default_factory=dict)  # question_id → chunks
    question_feedback: dict[str, int] = Field(default_factory=dict)  # question_id → last feedback

def record_usage(record: QARecord, session: SessionMemory):
    # Count each chunk once per question_id
    if record.question_id in session.question_links:
        return

    session.question_links[record.question_id] = set(record.used_chunk_ids)
    for cid in record.used_chunk_ids:
        belief = session.chunk_stats.setdefault(cid, ChunkBelief())
        belief.support_count += 1
        belief.last_used_in.append(cid)
    
    session.questions[record.question_id] = record.query

def apply_feedback(record: QARecord, session: SessionMemory):
    if record.user_feedback is None:
        return

    prev_feedback = session.question_feedback.get(record.question_id)
    if prev_feedback == record.user_feedback:
        return

    # Track which chunks were used for this question
    session.question_links[record.question_id] = set(record.used_chunk_ids)

    # If feedback changed, undo previous counts
    if prev_feedback is not None:
        for cid in record.used_chunk_ids:
            belief = session.chunk_stats.setdefault(cid, ChunkBelief())
            if prev_feedback > 0:
                belief.positive_feedback = max(0, belief.positive_feedback - 1)
            elif prev_feedback < 0:
                belief.negative_feedback = max(0, belief.negative_feedback - 1)

    for cid in record.used_chunk_ids:
        belief = session.chunk_stats.setdefault(cid, ChunkBelief())
        if record.user_feedback > 0:
            belief.positive_feedback += 1
        elif record.user_feedback < 0:
            belief.negative_feedback += 1

    session.question_feedback[record.question_id] = record.user_feedback


# SINGLE QUERY AGENT

retriever = IndexRetrieval()

## ------------------ RETRIEVAL ------------------ ##
class AgentState(BaseModel):
    # user intent
    query: str

    # retrieval control
    k: int = 5
    max_k: int = 20
    max_chunks: int = 30
    search_round: int = 1
    max_search_rounds: int = 3

    # retrieved data
    retrieved_chunks: List[str] = Field(default_factory=list)
    frontier_chunks: List[str] = Field(default_factory=list)
    visited_chunks: Set[str] = Field(default_factory=set)
    provenance: dict[str,str] = Field(default_factory=dict)

    # judgments
    sufficient: Optional[bool] = None
    sufficiency_reason: Optional[str] = None

    # memory (epistemic, session-scoped)
    remembered_chunks: Set[str] = Field(default_factory=set)
    rejected_chunks: Set[str] = Field(default_factory=set)

    # response
    answer: str = ""
    citations: dict[str, str] = None
    used_chunk_ids: List[str] = None

    # control flags
    stop: bool = False

def search_index(state: AgentState) -> AgentState:

    logger.info('------ SEARCH ------')

    logger.info('retrieved_chunks:')
    logger.info(state.retrieved_chunks)
    logger.info('frontier_chunks:')
    logger.info(state.frontier_chunks)
    logger.info('visited_chunks:')
    logger.info(state.visited_chunks)
    logger.info('provenance:')
    logger.info(state.provenance)

    current_ids = []
    neighbor_ids = []
    provenance = state.provenance.copy()
    # Add chunks from session memory on initialization
    if state.sufficient == None:
        logger.info('initialize with remembered chunks')
        for chunk_id in state.remembered_chunks:
            current_ids.append(chunk_id)
            provenance.update({chunk_id:'remembered'})
        logger.info(provenance)

    # Add neighbors of previously retrieved chunks
    for chunk_id in state.frontier_chunks:
        chunk_data = chunk_report(chunk_id)
        ngbrs = chunk_data.get('neighbors',[])
        for ngbr in ngbrs:
            ngbr_id = ngbr.get('id')
            ngbr_direction = f'{chunk_id}+{ngbr.get('direction','')}'
            if ngbr_id not in state.visited_chunks and ngbr_id not in state.rejected_chunks:
                neighbor_ids.append(ngbr_id)
                provenance.update({ngbr_id:ngbr_direction})

    chunks = retriever.search(state.query, k=state.k)
    
    for chunk in chunks:
        if isinstance(chunk, dict):
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in state.rejected_chunks:
                current_ids.append(chunk_id)
                provenance.update({chunk_id:'search'})
    # Do not add neighbors of neighbors
    front = [chunk_id for chunk_id in current_ids if chunk_id not in state.visited_chunks]

    for ngbr_id in neighbor_ids:
        if ngbr_id not in current_ids:
            current_ids.append(ngbr_id)

    result = {
        "retrieved_chunks": current_ids,
        "frontier_chunks": front,
        "visited_chunks": state.visited_chunks | set(current_ids),
        "provenance":provenance
    }

    logger.info('retrieved_chunks:')
    logger.info(result.get('retrieved_chunks'))
    logger.info('frontier_chunks:')
    logger.info(result.get('frontier_chunks'))
    logger.info('visited_chunks:')
    logger.info(result.get('visited_chunks'))
    logger.info('provenance:')
    logger.info(result.get('provenance'))

    logger.info('------ END SEARCH ------')

    return result


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

    response = LLM.generate_json(
        llm_judge_sufficiency_system_prompt, 
        prompt, 
        llm_judge_sufficiency_response_schema
    )

    verdict = SufficiencyVerdict(**response)
    return verdict

def judge_sufficiency(state: AgentState) -> AgentState:

    if len(state.retrieved_chunks) > state.max_chunks:
        chunk_ids = state.retrieved_chunks[:state.max_chunks]
    else:
        chunk_ids = state.retrieved_chunks

    verdict = llm_judge(
        query=state.query,
        chunk_ids=chunk_ids
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

    if len(state.retrieved_chunks) > state.max_chunks:
        return {"stop": True}

    if state.k < state.max_k:
        return {
            "k": min(state.k + 5, state.max_k),
            "search_round": state.search_round + 1
        }

    return {"stop": True}

## ------------------ ANSWER SYNTHESIS ------------------ #

def synthesize_answer(state: AgentState):
    logger.info('Synthesize Answer')


    token_limit = 30000
    token_estimate =0
    chunk_ids = []
    for chunk_id in state.retrieved_chunks:
        chunk_tokens = retriever.chunks[chunk_id]['token_estimate']
        if token_estimate + chunk_tokens <= token_limit:
            chunk_ids.append(chunk_id)
            token_estimate += chunk_tokens

    logger.info(f'Chunk Ids: {chunk_ids}')

    chunk_packet = {}
    for chunk_id in chunk_ids:
        chunk_data = chunk_report(chunk_id)
        chunk_packet[chunk_id] = chunk_data

    prompt = final_answer_user_prompt(state.query, list(chunk_packet.values()))

    response = LLM.generate_json(
        final_answer_system_prompt,
        prompt,
        final_answer_response_schema
    )

    answer = response.get("answer", "No answer generated.")
    citations = response.get("citations",{})
    used_chunk_ids = []
    provenance = {}
    result_citations = {}
    for reference in citations:
        key = reference['number']
        chunk_id = reference['chunk_id']
        chunk = chunk_packet.get(chunk_id,{})
        used_chunk_ids.append(chunk_id)
        chunk_provenance = state.provenance.get(chunk_id,'')
        provenance.update({chunk_id:chunk_provenance})
        ref = chunk.get('title') or ''
        ref += '\n' + (chunk.get('url') or '')
        result_citations[key] = ref

    return {
        "answer": answer,
        "citations": result_citations,
        "used_chunk_ids": used_chunk_ids,
        "provenance":provenance
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

## -------------------- HELPERS ---------------- ##

def chunk_report(chunk_id):

    chunk_data = {}
    chunk = retriever.chunks.get(chunk_id)
    paper_id = chunk.get('paper_id')
    paper = retriever.papers.get(paper_id)
    paper_meta = paper.get('meta')

    chunk_data['chunk_id'] = chunk_id        
    chunk_data['paper_id'] = paper_id 
    chunk_data['text'] = chunk.get('text')
    chunk_data['eqns'] = chunk.get('equations_raw',None)
    chunk_data['neighbors'] = chunk.get('neighbors',[])
    chunk_data['title'] = paper_meta.get('title')
    chunk_data['url'] = paper_meta.get('url')

    return chunk_data
