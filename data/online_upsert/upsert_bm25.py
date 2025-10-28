# upsert_bm25.py
import json, hashlib
from pathlib import Path
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, KEYWORD, NUMERIC

def open_or_create_whoosh(index_dir: str) -> "whoosh.index.Index":
    p = Path(index_dir)
    p.mkdir(parents=True, exist_ok=True)
    if index.exists_in(p):
        return index.open_dir(p)
    schema = Schema(
        chunk_id=ID(stored=True, unique=True),
        text=TEXT(stored=False),
        section=TEXT(stored=True),
        labels=KEYWORD(stored=True, commas=True, lowercase=True),
        paper_id=ID(stored=True),
        year=NUMERIC(stored=True)
    )
    return index.create_in(p, schema)

def upsert_md_bm25_from_jsonl(jsonl_path: str, bm_index_path: str) -> int:
    ix = open_or_create_whoosh(bm_index_path)
    n = 0
    with ix.writer() as w, open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            text = (rec.get("text") or "").strip()
            if not text: continue
            w.update_document(
                chunk_id=str(rec["chunk_id"]),
                text=text,
                section=rec.get("section",""),
                labels=",".join(rec.get("labels",[])),
                paper_id=rec.get("paper_id",""),
                year=int(rec.get("year",0)) if "year" in rec else 0
            )
            n += 1
    return n
