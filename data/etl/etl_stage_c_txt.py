"""Utilities for chunking plain-text files to JSON blocks suitable for LLMs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import config


# Prefer sentence-aware splits when we have to subdivide large paragraphs.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


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

def txt_collection_chunking(txt_files):

    txt_chunked_dir = config.TXT_CHUNKED_DIR
    chunked_files = []

    for txt_infile in txt_files:
        txt_infile = Path(txt_infile)
        txt_json_outfile = txt_chunked_dir / (txt_infile.with_suffix('.json')).name
        try:
            out_path = txt_file_chunking(txt_infile,txt_json_outfile)
            chunked_files.append(out_path)

        except Exception as e:
            print(f'Excption {e} for file {txt_infile}.')
    
    return chunked_files