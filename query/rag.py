# rag_runtime.py
from typing import List, Dict

def is_mathy(query: str) -> bool:
    q = query.lower()
    latex_tokens = ["\\lambda", "\\sum", "\\nabla", "\\alpha", "\\beta", "$", "eigenvalue", "spectral", "laplacian"]
    return any(tok in q for tok in latex_tokens)

def build_context_blocks(results: List[Dict], max_tokens: int = 1800) -> List[Dict]:
    """
    Trim and pack results into a token budget.
    Here we approximate tokens via len(text)//4 (same as earlier estimate).
    """
    kept, used = [], 0
    for result in results:
        txt = result.get("text") 
        est = max(1, len(txt) // 4)
        if used + est > max_tokens:
            continue
        kept.append({
            "chunk_id": result["chunk_id"],
            "paper_id": result.get("paper_id"),
            "section": result.get("section"),
            "text": txt,
            "equations_raw": result.get("equations_raw") or []
        })
        used += est
    return kept

def format_context_md(blocks: List[Dict]) -> str:
    """Render context as a citation-friendly block. Keep LaTeX as-is."""
    parts = []
    for b in blocks:
        header = f"[{b['chunk_id']}] {b.get('paper_id','')} — {b.get('section','')}"
        body = b["text"]
        # keep equations verbatim if present
        parts.append(f"{header}\n{body}\n")
    return "\n---\n".join(parts)

def build_prompt(question: str, context_str: str, math_mode: bool) -> Dict[str, str]:
    system = (
        "You are a careful technical assistant. Use ONLY the provided context to answer. "
        "If the context is insufficient, say so clearly. Always cite sources as [chunk_id]."
    )
    if math_mode:
        system += " Preserve LaTeX exactly; do not alter math notation."
    user = f"""
    QUESTION:
    {question}

    CONTEXT:
    {context_str}

    INSTRUCTIONS:
    - Answer concisely and accurately using the CONTEXT only.
    - Include inline citations to the most relevant chunk_ids like [paper123_s3_p2_c5].
    - If multiple steps or lemmas are used, cite each where used.
    - If the answer is uncertain with given CONTEXT, state what is missing.
    """.strip()
    return {"system": system, "user": user}

