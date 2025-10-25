import json
import random
from pydantic import BaseModel

class QueryReply(BaseModel):
    query: str
    complexity: int

def load_matches(path, keywords):
    matches = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed rows
            section = record.get("section", "")
            # any(keyword.lower() in section_path.lower() for keyword in keywords)
            section_path = record.get("section_path", "")
            keyword_in_sec = any(keyword.lower() in section.lower() + section_path.lower() for keyword in keywords)
            if keyword_in_sec:
                matches.append(record)
    return matches

def sample_wo_replacement(collection,num_elements):
    if num_elements > len(collection):
        sample = collection
    else:
        sample = random.sample(collection, num_elements)
    return sample

def text_to_query_prompt_builder(text):

    system = f"""
    You are a question generator for advanced STEM texts.

    GOAL:
    Create one thoughtful, conceptually clear question that could be answered by a student who studied the material summarized in the given text chunk — even if they no longer see that exact text.

    PROCESS:
    1. Read the text and extract its *main conceptual elements*, including:
       - the physical or mathematical system described,
       - key parameters and what varying them does,
       - contrasting viewpoints, interpretations, or domains (e.g., physics vs mathematics),
        - Ignore structural or organizational material (section summaries, theorem references, proof outlines, cross-references, etc.).
        - Focus only on scientific or mathematical statements that describe a system, property, phenomenon, or limitation.

2. Identify ONE intellectually central aspect of the text — this could be:
     • when a normally expected implication fails (e.g., positive Lyapunov exponent does not guarantee localization),
     • a specific regime or parameter case (e.g., criticality, localization, number theoretic properties),
     • when certain dynamical or spectral conditions change qualitative behavior,
     • when a model behaves differently under specific parameter regimes or approximations.
     • discussion of physical and mathematical viewpoints,
     • a mechanism or phenomenon the model aims to explain,

   Then, form a single question that:
     - explicitly names that conceptual situation using neutral descriptive phrases 
     - asks about a *non-trivial property, behavior, or contrast* within that setting,
     - asks the student to explain *why, compare, or predict* something within that context,
     - is specific enough that an expert reader could tell what concept it is testing,
     - could still be answered from a student’s notes without seeing the text itself.
       (e.g., “for ergodic Schrödinger operators with quasi-periodic dynamics,” 
             “when a system is well approximated by periodic transformations,” 
             “under conditions leading to a positive Lyapunov exponent”), 
     - remains concrete and content-anchored (avoid generic “a mathematical model” wording).

3. Phrase the question using domain-accurate terminology but without literal reuse of the source’s wording or symbols.
   Guidelines:
     - Allow: “ergodic Schrödinger operator,” “Lyapunov exponent,” “Anderson localization,” “periodic approximation.” etc.
     - Avoid: “this paper,” “the text,” “Equation (1),” or exact symbols (λ, α, θ).
     - When a symbol or name is unavoidable, replace it with a neutral descriptor (e.g., “a coupling parameter,” “the external field,” “the rotation parameter”).


    4. The question should read naturally and not start with vague scaffolds like “In this approach…”.

    5. If the text is purely definitional, return an empty query with complexity 0.
    OUTPUT FORMAT (JSON ONLY):
    {{
      "query": "<conceptually specific question>",
      "complexity": <integer 0–10>
    }}
    """.strip()

    user = f"""
    <CHUNK>
    {text}
    </CHUNK>
    Return ONLY the JSON object described above. No prose.
    """.strip()
    return {"system": system, "user": user}

def text_to_query_prompt(path,keywords,sample_size):
    evaluations = []
    matches = load_matches(path,keywords)
    sample = sample_wo_replacement(matches,sample_size)
    sample_texts = [ samp['text'] for samp in sample]
    for sample_text in sample_texts:
        prompt = text_to_query_prompt_builder(sample_text)
        record = {
            'text':sample_text,
            'prompt':prompt
        }
        evaluations.append(record)
    return evaluations

def text_to_query_llm(evaluations,client):
    for record in evaluations:
        response = client.responses.create(
            model="gpt-4o",
            input=[
                {"role": "system", "content": record['prompt']["system"]},
                {"role": "user", "content": record['prompt']["user"]},
            ],
            text={"format": {"type": "json_object"}},
        )
        response = json.loads(response.output_text)
        record['query'] = response['query']
        record['complexity'] = response['complexity']

def rag_judge_prompt_builder(original_text: str, query: str, rag_response: str):
    system = """
You are an impartial evaluator for advanced STEM content.

INPUTS:
- Query: what the user asked.
- Original Text: a local source chunk.
- RAG Response: an externally-sourced answer.

OBJECTIVE:
Score how good the RAG Response is *relative to* the Original Text. Reward added, correct, relevant information; penalize omissions, irrelevance, and contradictions.

EVALUATION DIMENSIONS (0–1 each):
1) Correctness vs Original (C): RAG facts do not contradict Original. If contradiction exists, subtract proportional penalty.
2) Coverage of Original (V): RAG includes all key points from Original that are needed to answer the Query.
3) Enrichment Beyond Original (E): RAG adds correct, relevant details not present in Original (definitions, conditions, implications, examples, references to known results).
4) Specificity & Utility (S): RAG gives concrete, technically meaningful statements (not vague restatements).
5) Focus/Relevance to Query (R): RAG stays on-topic and addresses the Query directly.

CONTRADICTION HANDLING:
- If a clear, material conflict with Original is present, mark `has_contradiction=true`, summarize it, and cap C ≤ 0.4.
- If RAG asserts novel claims, require them to be coherent with Original; otherwise treat as partial conflict.

OVERALL SCORE (0–100):
score = round( 100 * (0.30*C + 0.25*V + 0.25*E + 0.10*S + 0.10*R) )

GRADE BANDS:
- 90–100: Much better than Original (adds strong, correct insight; complete and precise).
- 75–89: Better than Original (some valuable, correct additions; minor gaps).
- 60–74: Roughly comparable (little added value or small gaps).
- 40–59: Worse than Original (misses key points or weak specificity).
- 0–39: Poor (off-topic, incorrect, or contradicts Original).

OUTPUT FORMAT (JSON ONLY):
{
  "score": 0-100,
  "correctness":0-1, 
  "coverage":0-1, 
  "enrichment":0-1, 
  "specificity":0-1, 
  "relevance":0-1,
  "band": "excellent|good|fair|poor|very_poor", 
  "original_key_points": ["...","..."],
  "improvements_added_by_rag": ["..."],
  "issues_found": ["..."],
  "contradictions": [
    {"claim_in_rag":"...", "conflicts_with_original":"...", "why_it_matters":"..."}
  ],
  "verdict": "1–3 sentence justification"
}

RULES:
- Extract 2–6 “original_key_points” first (short bullets).
- Be conservative: if unsure a RAG claim is correct, do not reward E; flag under issues.
- No chain-of-thought; return only the JSON object."""
    user = f"""
Query:
{query}

Original Text:
{original_text}

RAG Response:
{rag_response}

TASK:
- Evaluate the RAG Response to the query against the content of the orginal text.
- Return ONLY the JSON object described above.
"""
    return system, user