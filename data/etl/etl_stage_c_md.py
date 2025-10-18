import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any
import config

LABEL_RE = re.compile(r'\\label\{([^}]+)\}')
REF_RE   = re.compile(r'\\(?:eqref|ref)\{([^}]+)\}')
HTML_REF_RE = re.compile(r"data-reference\s*=\s*[\"']([^\"'\s]+)[\"']", re.IGNORECASE)
HTML_ID_RE = re.compile(r"<(?:div|span|section|h[1-6]|p|table|figure|figcaption)[^>]*\bid\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)

MATH_INLINE_RE = re.compile(r"\$(?!\s)(.+?)(?<!\s)\$", re.DOTALL)  # $...$
MATH_DISPLAY_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)          # $$...$$
MATH_BRACKET_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)          # \[...\]
MATH_ENV_RE = re.compile(
    r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?)\}(.+?)\\end\{\1\}",
    re.DOTALL
)

def _extract_labels_refs(text: str):
    labels = set(LABEL_RE.findall(text))
    labels.update(HTML_ID_RE.findall(text))
    refs = set(REF_RE.findall(text))
    refs.update(HTML_REF_RE.findall(text))
    return sorted(labels), sorted(refs)

def _is_heading(line: str) -> bool:
    # ATX headings only (#..###### ). Fine for Pandoc's default md.
    s = line.lstrip()
    return s.startswith("#") and not s.startswith("#######!")

def _heading_level(line: str) -> int:
    return len(line.lstrip().split()[0]) if _is_heading(line) else 0

def _normalize_section_path(headings_stack: List[str]) -> str:
    return " / ".join(h.strip() for h in headings_stack if h.strip())

def md_file_chunking(input_md_filepath, output_json_filepath):
    """
    chunk md file and output chunks into json file

    Strategy:
      - Split by headings, but within a section we create sub-chunks of ~max_chars,
        preferring to break on blank lines (paragraph boundaries).
      - Do not split inside fenced code blocks (``` ... ```) or display-math blocks ($$ ... $$).
      - Record LaTeX labels/refs found in each chunk for graph building later.
    Output JSON schema (array of objects):
      {
        "id": str,
        "file": str,
        "section_path": str,  # e.g., "1 Introduction / 1.1 Background"
        "start_line": int,    # 1-based
        "end_line": int,      # inclusive
        "text": str,
        "labels": [str],
        "refs": [str],
        "meta": {"heading": {"level": int, "title": str}}
      }
    """
    in_path = Path(input_md_filepath)
    out_path = Path(output_json_filepath)
    lines = in_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Tunables
    max_chars = 1800          # target chunk size
    hard_max_chars = 2400     # hard ceiling if no good break found
    para_overlap = 1          # carry last N paragraphs into next chunk (context)
    min_chunk_chars = 200

    chunks: List[Dict[str, Any]] = []
    headings_stack: List[str] = []
    current_section_lines: List[str] = []
    current_section_start_line = 1
    current_heading_title = ""
    current_heading_level = 0

    in_code = False
    in_math = False
    fence_delim = ""
    math_delim = ""

    def flush_subchunks(section_text_lines: List[str], section_start_line: int,
                        section_heading_level: int, section_heading_title: str):
        """
        Break a section's lines into paragraph-aware subchunks respecting max_chars.
        """
        nonlocal chunks
        if not section_text_lines:
            return

        # Build paragraph units while respecting fences
        paras: List[Dict[str, Any]] = []  # {text:str, start:int, end:int}
        buf: List[str] = []
        start_line = section_start_line
        in_code_local = False
        in_math_local = False
        fence_local = ""
        math_local = ""

        def push_para(end_line: int):
            if buf:
                paras.append({"text": "\n".join(buf).rstrip(), "start": start_line, "end": end_line})
        
        for idx, line in enumerate(section_text_lines, start=section_start_line):
            stripped = line.strip()

            # toggle code fence
            if not in_math_local and stripped.startswith("```"):
                if not in_code_local:
                    in_code_local = True
                    fence_local = stripped  # remember exact fence
                else:
                    in_code_local = False
                    fence_local = ""
                buf.append(line)
                continue

            # toggle display math ($$ blocks; ignore inline $...$)
            if not in_code_local and stripped.startswith("$$"):
                if not in_math_local:
                    in_math_local = True
                    math_local = "$$"
                else:
                    in_math_local = False
                    math_local = ""
                buf.append(line)
                continue

            buf.append(line)

            # paragraph boundary if blank line and not inside a block
            if (not in_code_local and not in_math_local) and stripped == "":
                push_para(idx)
                buf = []
                start_line = idx + 1

        # tail
        push_para(section_start_line + len(section_text_lines) - 1)

        # Merge paras into chunks by size
        i = 0
        chunk_idx_within_section = 0
        while i < len(paras):
            carry = max(0, i - para_overlap)  # for overlap
            j = i
            text_parts = []
            start = paras[i]["start"]
            end = paras[i]["end"]
            size = 0

            while j < len(paras):
                ptxt = paras[j]["text"]
                add_len = len(ptxt) + 1  # newline
                if size + add_len > max_chars and (size >= min_chunk_chars):
                    break
                if size + add_len > hard_max_chars and size > 0:
                    break
                text_parts.append(ptxt)
                size += add_len
                end = paras[j]["end"]
                j += 1

            # Prepend overlap paragraphs if any (read-only context)
            if carry < i:
                overlap_text = "\n\n".join(p["text"] for p in paras[carry:i])
                if overlap_text.strip():
                    text_parts = [overlap_text, *text_parts]

            text = "\n\n".join(tp for tp in text_parts if tp is not None).strip()
            labels, refs = _extract_labels_refs(text)
            equations = _extract_equations(text)
            section_path = _normalize_section_path(headings_stack)
            chunk_id = f"{in_path.stem}::L{start}-{end}::s{chunk_idx_within_section}"

            chunks.append({
                "id": chunk_id,
                "file": str(in_path),
                "section_path": section_path,
                "start_line": start,
                "end_line": end,
                "text": text,
                "labels": labels,
                "refs": refs,
                "equations_raw": equations,
                "equation_count": len(equations),
                "meta": {
                    "heading": {"level": section_heading_level, "title": section_heading_title}
                }
            })

            chunk_idx_within_section += 1
            i = j

    # Walk lines, split into sections on headings (but don’t break inside fences/math)
    i = 0
    while i < len(lines):
        line_no = i + 1
        line = lines[i]
        stripped = line.strip()

        # code fence toggle
        if not in_math and stripped.startswith("```"):
            if not in_code:
                in_code = True
                fence_delim = stripped
            else:
                in_code = False
                fence_delim = ""
            current_section_lines.append(line)
            i += 1
            continue

        # display math fence toggle
        if not in_code and stripped.startswith("$$"):
            if not in_math:
                in_math = True
                math_delim = "$$"
            else:
                in_math = False
                math_delim = ""
            current_section_lines.append(line)
            i += 1
            continue

        # Heading (only if not inside code/math)
        if not in_code and not in_math and _is_heading(line):
            # Flush previous section (including its heading context)
            flush_subchunks(current_section_lines, current_section_start_line,
                            current_heading_level, current_heading_title)
            current_section_lines = []
            current_section_start_line = line_no

            # Update heading stack
            level = _heading_level(line)
            title = line.lstrip().split(' ', 1)[1] if ' ' in line.lstrip() else ""
            # Reduce stack to level-1
            while len(headings_stack) >= level:
                headings_stack.pop()
            headings_stack.append(title)
            current_heading_level = level
            current_heading_title = title

            current_section_lines.append(line)  # keep heading with its section
            i += 1
            continue

        # normal line
        current_section_lines.append(line)
        i += 1

    # Flush tail section
    flush_subchunks(current_section_lines, current_section_start_line,
                    current_heading_level, current_heading_title)

    # Populate neighbor metadata after all chunks are assembled so we retain adjacency
    # and can augment with cross references between labeled equations and referring text.
    def _append_neighbor(neighbor_list: List[Dict[str, str]], neighbor: Dict[str, str]):
        # Avoid duplicate neighbor-direction pairs.
        if not any(n["id"] == neighbor["id"] and n["direction"] == neighbor["direction"] for n in neighbor_list):
            neighbor_list.append(neighbor)

    for chunk in chunks:
        chunk["neighbors"] = []

    for idx, chunk in enumerate(chunks):
        if idx > 0:
            prev_chunk = chunks[idx - 1]
            _append_neighbor(chunk["neighbors"], {
                "id": prev_chunk["id"],
                "direction": "previous"
            })

        if idx + 1 < len(chunks):
            next_chunk = chunks[idx + 1]
            _append_neighbor(chunk["neighbors"], {
                "id": next_chunk["id"],
                "direction": "next"
            })

    # Build mapping from labels to chunks for reference resolution.
    label_to_chunk_indices: Dict[str, List[int]] = {}
    for idx, chunk in enumerate(chunks):
        for label in chunk.get("labels", []):
            label_to_chunk_indices.setdefault(label, []).append(idx)

    # Link referencing chunks to labeled chunks and reciprocate as comments.
    for idx, chunk in enumerate(chunks):
        for ref in chunk.get("refs", []):
            for target_idx in label_to_chunk_indices.get(ref, []):
                if target_idx == idx:
                    continue
                target_chunk = chunks[target_idx]
                _append_neighbor(chunk["neighbors"], {
                    "id": target_chunk["id"],
                    "direction": "reference"
                })
                _append_neighbor(target_chunk["neighbors"], {
                    "id": chunk["id"],
                    "direction": "comment"
                })

    # Write JSON array
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return str(out_path)


# --- Field helpers ------------------------------------------------------------


def _derive_paper_id(rec: Dict[str, Any]) -> str:
    """Extract paper_id from id or file name."""
    rid = rec.get("id") or ""
    if "::" in rid:
        return rid.split("::", 1)[0]
    fpath = rec.get("file") or ""
    if fpath:
        return Path(fpath).stem
    return ""

def _choose_section(rec: Dict[str, Any]) -> str:
    """Pick section name from heading or section_path."""
    meta = rec.get("meta") or {}
    heading = (meta.get("heading") or {}).get("title")
    if heading and isinstance(heading, str) and heading.strip():
        return heading.strip()
    sec = rec.get("section_path") or ""
    return str(sec).strip()

def _extract_equations(text: str):
    """Return labeled LaTeX math snippets in text."""
    eqs = []
    eqs += [m.group(1).strip() for m in MATH_DISPLAY_RE.finditer(text or "")]
    eqs += [m.group(1).strip() for m in MATH_BRACKET_RE.finditer(text or "")]
    eqs += [m.group(2).strip() for m in MATH_ENV_RE.finditer(text or "")]
    eqs += [m.group(1).strip() for m in MATH_INLINE_RE.finditer(text or "")]
    eqs = [e for e in eqs if LABEL_RE.search(e)]
    # Deduplicate while preserving order
    seen, uniq = set(), []
    for e in eqs:
        k = re.sub(r"\s+", " ", e)
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    return uniq

def _stable_chunk_id(rec: Dict[str, Any], fallback_index: int) -> str:
    """Return a stable chunk ID."""
    if rec.get("id"):
        return str(rec["id"])
    pid = _derive_paper_id(rec) or "paper"
    return f"{pid}::chunk{fallback_index}"

def _estimate_tokens(text: str) -> int:
    """Crude token estimate for chunk size QA."""
    chars = len(text)
    words = max(1, len(text.split()))
    est = max(1, min(words * 2, chars // 4))
    return est

def _normalize_record(rec: Dict[str, Any], fallback_index: int) -> Dict[str, Any]:
    """Normalize one chunk dict."""
    text = (rec.get("text") or "").strip()
    section = _choose_section(rec)
    pid = _derive_paper_id(rec)
    chunk_id = _stable_chunk_id(rec, fallback_index)
    equations = _extract_equations(text)

    return {
        "chunk_id": chunk_id,
        "paper_id": pid,
        "source_file": rec.get("file") or "",
        "section": section,
        "labels": rec.get("labels") or [],
        "refs": rec.get("refs") or [],
        "neighbors": rec.get("neighbors") or [],
        "start_line": rec.get("start_line"),
        "end_line": rec.get("end_line"),
        "text": text,
        "text_len": len(text),
        "token_estimate": _estimate_tokens(text),
        "equations_raw": equations,
        "equation_count": len(equations),
        "added_at": int(time.time()),
        "version": "ppmdc-0.1",
    }


def _collect_json_payloads(paths: List[Path]) -> List[Dict[str, Any]]:
    """Load and flatten JSON payloads coming from chunk files."""
    records: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            records.extend(payload)
        elif isinstance(payload, dict):
            records.append(payload)
    return records


def _normalize_records(records: List[Dict[str, Any]], drop_empty: bool) -> List[Dict[str, Any]]:
    """Normalize raw chunk dicts and optionally drop empty text entries."""
    normalized: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records):
        norm = _normalize_record(rec, fallback_index=idx)
        if drop_empty and not norm["text"]:
            continue
        normalized.append(norm)
    return normalized


def _write_jsonl(records: List[Dict[str, Any]], out_path: Path) -> None:
    """Write normalized records into JSONL format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as w:
        for rec in records:
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

# --- Main entry point ---------------------------------------------------------

def post_process_md_chunking(list_of_json_paths: List[str],
                             combined_output_jsonl: str,
                             drop_empty: bool = True) -> Dict[str, Any]:
    """
    Merge and normalize chunk JSON files into one JSONL.

    Args:
        list_of_json_paths: list of paths to input JSON files.
        combined_output_jsonl: output JSONL path.
        drop_empty: if True, skip chunks with empty text.

    Returns:
        dict summary with counts, paper IDs, and normalized records.
    """
    out_path = Path(combined_output_jsonl)
    input_paths = [Path(p) for p in list_of_json_paths]
    raw_records = _collect_json_payloads(input_paths)
    normalized_records = _normalize_records(raw_records, drop_empty=drop_empty)

    _write_jsonl(normalized_records, out_path)

    paper_ids = {rec["paper_id"] for rec in normalized_records if rec.get("paper_id")}

    return {
        "md_json_files": len(list_of_json_paths),
        "records_seen": len(raw_records),
        "records_written": len(normalized_records),
        "unique_paper_ids": sorted(paper_ids),
        "output_jsonl": str(out_path),
        "normalized_records": normalized_records,
    }



def md_collection_chunking(md_files):

    md_chunked_dir = config.MD_CHUNKED_DIR
    md_chunked_dir.mkdir(parents=True, exist_ok=True)
    chunked_files = {}

    md_jsonl = config.MD_JSONL

    for arxiv_id, md_infile in md_files.items():
        md_infile = Path(md_infile)
        md_json_outfile = md_chunked_dir / (md_infile.with_suffix('.json')).name
        try:
            out_path = md_file_chunking(md_infile,md_json_outfile)
            chunked_files[arxiv_id] = out_path

        except Exception as e:
            print(f'Excption {e} for file {md_infile}.')
    
    aggregated_chunk_files = sorted(md_chunked_dir.glob("*.json"))
    out = {}
    out['md_details'] = post_process_md_chunking([str(p) for p in aggregated_chunk_files], md_jsonl)
    out["md_chunk_files"] = chunked_files

    return out
