"""Utilities for chunking plain-text files to JSON blocks suitable for LLMs."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import config


# Prefer sentence-aware splits when we have to subdivide large paragraphs.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

ARXIV_ID_RE = re.compile(r"\barXiv:(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})\b", re.I)
ARXIV_URL_RE = re.compile(r"https?://arxiv\.org/abs/([^\s]+)", re.I)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"https?://\S+")
SECTION_CAND_RE = re.compile(
    r"^(abstract|introduction|preliminaries|related work|results|proofs?|"
    r"conclusion|acknowledg(e)?ments?|references?)\s*$", re.I
)

# Heuristic indicators of math stripped by detex
MATH_LOSS_PATTERNS = [
    r"\(\s*\)", r"\[\s*\]", r"\{\s*\}",
    r"^\*{1,3}\s*$",
    r"^eq(\.|:)", r"eq:\w+",
    r"\\(alpha|beta|gamma|lambda|Delta|nabla)\b",
    r"\b[A-Z]\w*\s*=\s*$",
]
MATH_LOSS_RE = re.compile("|".join(MATH_LOSS_PATTERNS), re.I | re.M)





def _split_long_paragraph(text: str, max_chars: int) -> List[str]:
    """Break a long paragraph into <=max_chars chunks.

    We first try sentence boundaries, and fall back to whitespace splits if a
    single sentence still exceeds the budget. The returning strings have their
    internal whitespace normalised to single spaces.
    """

    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return [normalized]

    sentences = _SENTENCE_BOUNDARY_RE.split(normalized) or [normalized]
    segments: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        add_len = len(sentence) + (1 if current else 0)
        if current and current_len + add_len > max_chars:
            segments.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += add_len

    if current:
        segments.append(" ".join(current))

    # Sentence wrapping might still produce segments above the limit if a single
    # sentence is enormous. In that case we split on whitespace.
    refined: List[str] = []
    for segment in segments:
        if len(segment) <= max_chars:
            refined.append(segment)
            continue
        words = segment.split()
        bucket: List[str] = []
        bucket_len = 0
        for word in words:
            add = len(word) + (1 if bucket else 0)
            if bucket and bucket_len + add > max_chars:
                refined.append(" ".join(bucket))
                bucket = [word]
                bucket_len = len(word)
            else:
                bucket.append(word)
                bucket_len += add
        if bucket:
            refined.append(" ".join(bucket))

    return refined or [normalized]


def txt_file_chunking(input_txt_filepath, output_json_filepath):
    """Chunk a plain-text file into JSON-formatted sections for downstream LLMs."""

    in_path = Path(input_txt_filepath)
    out_path = Path(output_json_filepath)

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    raw_lines = in_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    paragraphs: List[Dict[str, Any]] = []
    buffer: List[tuple[int, str]] = []
    start_line = 1

    for line_no, raw_line in enumerate(raw_lines, start=1):
        stripped = raw_line.strip()
        if stripped:
            if not buffer:
                start_line = line_no
            buffer.append((line_no, stripped))
        else:
            if buffer:
                para_text = "\n".join(text for _, text in buffer).strip()
                if para_text:
                    alnum = re.sub(r"[^0-9A-Za-z]+", "", para_text)
                    if len(alnum) > 2 or " " in para_text:
                        paragraphs.append(
                            {
                                "start": start_line,
                                "end": buffer[-1][0],
                                "text": para_text,
                            }
                        )
                buffer = []

    if buffer:
        para_text = "\n".join(text for _, text in buffer).strip()
        if para_text:
            alnum = re.sub(r"[^0-9A-Za-z]+", "", para_text)
            if len(alnum) > 2 or " " in para_text:
                paragraphs.append(
                    {
                        "start": start_line,
                        "end": buffer[-1][0],
                        "text": para_text,
                    }
                )

    if not paragraphs:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("[]\n", encoding="utf-8")
        return str(out_path)

    # Parameters roughly aligned with the Markdown chunker.
    max_chars = 1800
    hard_max_chars = 2400
    para_overlap = 1
    min_chunk_chars = 200

    expanded: List[Dict[str, Any]] = []
    for para in paragraphs:
        text = para["text"]
        if len(text) <= hard_max_chars:
            expanded.append(para)
            continue
        for segment in _split_long_paragraph(text, max_chars):
            if not segment.strip():
                continue
            expanded.append(
                {
                    "start": para["start"],
                    "end": para["end"],
                    "text": segment,
                }
            )

    chunks: List[Dict[str, Any]] = []
    i = 0
    chunk_idx = 0
    total = len(expanded)
    while i < total:
        carry_idx = max(0, i - para_overlap)
        j = i
        text_parts: List[str] = []
        start = expanded[i]["start"]
        end = expanded[i]["end"]
        size = 0

        while j < total:
            para_text = expanded[j]["text"]
            add_len = len(para_text) + (2 if text_parts else 0)
            if text_parts and size + add_len > max_chars and size >= min_chunk_chars:
                break
            if text_parts and size + add_len > hard_max_chars:
                break

            text_parts.append(para_text)
            size += add_len
            end = expanded[j]["end"]
            j += 1

        if carry_idx < i:
            overlap_text = "\n\n".join(expanded[k]["text"] for k in range(carry_idx, i))
            if overlap_text.strip():
                text_parts = [overlap_text, *text_parts]

        chunk_text = "\n\n".join(text_parts).strip()
        if not chunk_text:
            i = j
            continue

        chunk_id = f"{in_path.stem}::L{start}-{end}::c{chunk_idx}"
        chunks.append(
            {
                "id": chunk_id,
                "file": str(in_path),
                "section_path": "",
                "start_line": start,
                "end_line": end,
                "text": chunk_text,
                "labels": [],
                "refs": [],
                "meta": {"source": "plain_text"},
            }
        )

        chunk_idx += 1
        i = j if j > i else i + 1

    # Link sequential neighbours for navigation context.
    for idx, chunk in enumerate(chunks):
        neighbors: List[Dict[str, str]] = []
        if idx > 0:
            neighbors.append({"id": chunks[idx - 1]["id"], "direction": "previous"})
        if idx + 1 < len(chunks):
            neighbors.append({"id": chunks[idx + 1]["id"], "direction": "next"})
        chunk["neighbors"] = neighbors

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)

    return str(out_path)


# --- Helpers ------------------------------------------------------------------

def _derive_paper_id(rec: Dict[str, Any]) -> str:
    """Extract paper_id from record id or filename."""
    rid = rec.get("id") or ""
    if "::" in rid:
        return rid.split("::", 1)[0]
    fpath = rec.get("file") or ""
    if fpath:
        return Path(fpath).stem
    return ""

def _estimate_tokens(text: str) -> int:
    """Crude token estimate for QA."""
    chars = len(text)
    words = max(1, len(text.split()))
    return max(1, min(words * 2, chars // 4))

def _guess_chunk_type(text: str, section_path: str) -> str:
    """Roughly classify chunk content."""
    t = (text or "").strip()
    if not t:
        return "empty"
    first = t.splitlines()[0].strip()
    if SECTION_CAND_RE.match(first):
        name = SECTION_CAND_RE.match(first).group(1).lower()
        if name.startswith("reference"):
            return "references_header"
        return f"section_header:{name}"
    if "abbrv" in t.lower() or "bibliography" in t.lower() or re.search(r"^\s*\[\s*\d+\s*\]", t, re.M):
        return "references_block"
    if EMAIL_RE.search(t) or "university" in t.lower() or "department" in t.lower():
        return "front_matter"
    if re.search(r"\b(theorem|lemma|proposition|corollary|definition)\b", t, re.I):
        return "math_body"
    if re.search(r"\bfigure\b|\btable\b", t, re.I) or re.search(r"^\s*-?\d+(\.\d+)?\s*$", t, re.M):
        return "figure_or_numeric"
    return "body"

def _detect_math_loss(text: str) -> bool:
    """Flag whether the text likely lost equations during detex."""
    if not text or not text.strip():
        return False
    if MATH_LOSS_RE.search(text):
        return True
    if re.search(r"[=+\-*/]\s*$", text, re.M):
        return True
    return False

def _harvest_refs(text: str) -> Dict[str, Any]:
    """Collect arXiv IDs, emails, and URLs."""
    arxiv_ids = set(m.group(1) for m in ARXIV_ID_RE.finditer(text or ""))
    arxiv_ids |= set(m.group(1) for m in ARXIV_URL_RE.finditer(text or ""))
    emails = set(m.group(0) for m in EMAIL_RE.finditer(text or ""))
    urls = set(m.group(0) for m in URL_RE.finditer(text or ""))
    return {
        "arxiv_ids": sorted(arxiv_ids),
        "emails": sorted(emails),
        "urls": sorted(urls),
    }

def _normalize_txt_record(rec: Dict[str, Any], fallback_index: int) -> Dict[str, Any]:
    """Normalize one TXT chunk record."""
    text = (rec.get("text") or "").strip()
    pid = _derive_paper_id(rec)
    chunk_id = rec.get("id") or f"{pid}::chunk{fallback_index}"
    chunk_type = _guess_chunk_type(text, rec.get("section_path", ""))
    harvested = _harvest_refs(text)
    return {
        "chunk_id": chunk_id,
        "paper_id": pid,
        "source_file": rec.get("file") or "",
        "section_path": rec.get("section_path", ""),
        "start_line": rec.get("start_line"),
        "end_line": rec.get("end_line"),
        "chunk_type": chunk_type,
        "text": text,
        "text_len": len(text),
        "token_estimate": _estimate_tokens(text),
        "has_math_loss": _detect_math_loss(text),
        "labels": rec.get("labels") or [],
        "neighbors": rec.get("neighbors") or [],
        "meta": (rec.get("meta") or {}) | {"source_kind": "plain_text"},
        "harvest": harvested,
        "added_at": int(time.time()),
        "version": "pptxt-0.1",
    }

# --- Main function ------------------------------------------------------------

def post_process_txt_chunking(list_of_json_paths: List[str],
                              combined_output_jsonl: str,
                              drop_empty: bool = True) -> Dict[str, Any]:
    """
    Merge and normalize TXT (detex) chunk JSONs into one JSONL.

    Args:
        list_of_json_paths: list of input JSON files.
        combined_output_jsonl: output JSONL path.
        drop_empty: skip empty text chunks if True.

    Returns:
        dict summary with counts and paper IDs.
    """
    out_path = Path(combined_output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_in = 0
    total_out = 0
    paper_ids = set()

    with out_path.open("w", encoding="utf-8") as w:
        for path in list_of_json_paths:
            p = Path(path)
            if not p.exists():
                continue
            with p.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            records = payload if isinstance(payload, list) else [payload]
            for i, rec in enumerate(records):
                total_in += 1
                norm = _normalize_txt_record(rec, i)
                if drop_empty and not norm["text"]:
                    continue
                paper_ids.add(norm["paper_id"] or "")
                w.write(json.dumps(norm, ensure_ascii=False) + "\n")
                total_out += 1

    return {
        "input_files": len(list_of_json_paths),
        "records_seen": total_in,
        "records_written": total_out,
        "unique_paper_ids": sorted(pid for pid in paper_ids if pid),
        "output_jsonl": str(out_path),
    }



def txt_collection_chunking(txt_files):

    txt_chunked_dir = config.TXT_CHUNKED_DIR
    chunked_files = []

    txt_jsonl = config.TXT_JSONL

    for txt_infile in txt_files:
        txt_infile = Path(txt_infile)
        txt_json_outfile = txt_chunked_dir / (txt_infile.with_suffix('.json')).name
        try:
            out_path = txt_file_chunking(txt_infile,txt_json_outfile)
            chunked_files.append(out_path)

        except Exception as e:
            print(f'Excption {e} for file {txt_infile}.')
    
    txt_details = post_process_txt_chunking(chunked_files,txt_jsonl)

    return txt_details