

## Stage A

Pull data from arXiv.

### Inputs
- Query inputs: `phrases` (list[str]) and `categories` (list[str]) used by `build_arxiv_query()` to construct an arXiv advanced query in the form `(all:"phrase1" OR ...) AND (cat:cat1 OR ...)`.
- Retrieval limits: `max_results` (int) passed to `arxiv_client_search()` to control pagination/batch count.
- External services:
  - arXiv API via `arxiv` Python client for search metadata + source downloads.
  - Semantic Scholar Graph API for citations/references per arXiv ID.
- Configured paths (from `etl/config.py`): `TAR_DIR`, `TAR_EXTRACT_DIR`, and `STAGE_A` output file location under `data/data_etl`.

### Outputs
- Source archives downloaded to `data/data_etl/tar/{arxiv_id}.tar.gz`.
- Extracted LaTeX sources under `data/data_etl/latex_raw/{arxiv_id}/`.
  - If the source is a single gzip file, writes `{arxiv_id}.tex` inside that directory.
- Stage A report JSON written to `data/data_etl/stages/stage_a.json` with:
  - `arviv_query` (string; note the current key spelling).
  - `papers` dict keyed by `arxiv_id`, each containing:
    - `meta` serialized `PaperMeta` (arxiv_id, base_id, sanitized_id, version, title, categories, authors, published/updated dates, url, gzip, summary, comment, citation_list, reference_list, extract).
    - `extract` (bool indicating download success).
    - `latex_dir` (path or null).
- In-memory return from `arxiv_extract()` mirrors the `papers` structure above (dict of per-paper records).

### Norms
- arXiv search pagination: page_size=25, batch_size=50, delay_seconds=1, num_retries=3; sleeps 3s between batches and 0.5s per record to pace requests.
- De-duplication: results are keyed by `arxiv_id = get_short_id().replace("/", "_")`; duplicates are skipped.
- File naming:
  - Downloaded archive name: `{arxiv_id}.tar.gz`.
  - `sanitized_id` derived from entry_id/version; used in metadata, not filenames here.
- Extraction safety:
  - Tar extraction enforces path traversal checks, skips symlinks/links, and creates parent dirs.
  - If the archive is not a tar or gzip file, extraction returns `None`.
- Semantic Scholar calls use `fields=title,year,venue,externalIds,authors,url`, `User-Agent: refs-fetch/1.0`, and 20s timeouts.
- Updated date logic: `updated_date` uses arXiv `updated` if present, otherwise `published`.


## Stage B

Transform latex

### Inputs
- `papers` records from Stage A (specifically `latex_dir` paths per arXiv ID).
- Configured paths (from `etl/config.py`): `LATEX_FILTER_DIR`, `MD_VERSION_DIR`, `TEXT_VERSION_DIR`, and `STAGE_B`.
- Optional external binaries if available on PATH:
  - `latexpand` for full TeX expansion (best).
  - `pandoc` for Markdown and/or plain-text conversion.
  - `detex` for plain-text conversion.

### Outputs
- Cleaned/combined LaTeX written to `data/data_etl/latex_final/{arxiv_id}.tex`.
- Markdown output written to `data/data_etl/full_markdown/{arxiv_id}.md` when `pandoc` is available.
- Plain-text output written to `data/data_etl/full_text/{arxiv_id}.txt` via `detex`, `pandoc -t plain`, or naive fallback.
- Stage B report JSON written to `data/data_etl/stages/stage_b.json` containing the Stage A payload plus:
  - `combined_latex_path` per paper.
  - `md_full_path` and `txt_full_path` per paper when conversions succeed.

### Norms
- Main TeX selection: chooses the highest-scoring `.tex` based on `\documentclass`, `\begin{document}`, `\title`, and filename hints (`main*`, `arxiv*`). If no good candidate, concatenates all `.tex` in alpha order.
- Include handling: attempts `latexpand` first; if unavailable/fails, inlines `\input`, `\include`, and `\subfile` recursively with cycle protection.
- Cleaning rules:
  - Strips LaTeX comments (`%`) while preserving escaped `\%`.
  - Drops environments: `figure`, `figure*`, `table`, `table*`, `tikzpicture`, `axis`.
  - Removes `\includegraphics` commands (with optional args).
- Markdown conversion: `pandoc -f latex -t gfm --wrap none`.
- Text conversion: prefers `detex`, then `pandoc -t plain`, then a naive regex-based stripper; always writes a `.txt` if any fallback succeeds.


## Stage C

Chunking Docs

### Inputs
- Stage B outputs per paper:
  - `md_full_path` for Markdown chunking.
  - `txt_full_path` for plain-text chunking.
- Configured paths (from `etl/config.py`): `MD_CHUNKED_DIR`, `TXT_CHUNKED_DIR`, `MD_JSONL`, `TXT_JSONL`.
- Logger writes to `log/logs/etl.log` via `DEFAULT_LOG_DIR`.

### Outputs
- Per-paper chunk JSON files:
  - Markdown chunks in `data/data_etl/md_chunked/{arxiv_id}.json`.
  - Text chunks in `data/data_etl/txt_chunked/{arxiv_id}.json`.
- Aggregated JSONL:
  - Markdown `data/data_etl/md_data.jsonl`.
  - Text `data/data_etl/txt_data.jsonl`.
- Normalized records returned to Stage C caller:
  - `normed_records` dict keyed by `paper_id`, then `chunk_id`, for both md and txt paths.
  - Summary stats: counts of files/records and `unique_paper_ids`.

### Norms
- Markdown chunking (`stg_c_md_chunking.py`):
  - Splits by headings (ATX `#` style) into sections; inside each section, builds subchunks targeting ~1800 chars (hard max 2400) with 1-paragraph overlap and 200-char minimum.
  - Avoids splitting inside fenced code blocks and display-math blocks (`$$...$$`).
  - Extracts LaTeX labels/refs and HTML IDs/refs, captures labeled equations, and builds `neighbors` links:
    - `previous`/`next` adjacency, plus `reference`/`comment` cross-links for label↔ref matches.
  - Chunk IDs follow `{paper_id}::L{start}-{end}::s{n}` with 1-based line ranges.
  - Normalized record fields include `section`, `labels`, `refs`, `equations_raw`, `token_estimate`, and `version: "ppmdc-0.1"`.
- Text chunking (`stg_c_txt_chunking.py`):
  - Parses paragraphs, drops ultra-short noise, and splits long paragraphs on sentence boundaries to respect the same size budgets (1800/2400/200) with 1-paragraph overlap.
  - Infers section headers from common headings (Abstract, Introduction, Results, etc.) and uses first-line fallback.
  - Adds sequential `previous`/`next` neighbors only (no cross-reference linking).
  - Harvests `arxiv_ids`, `emails`, and `urls`, and flags `has_math_loss` using heuristic patterns.
  - Normalized record fields include `chunk_type`, `harvest`, `token_estimate`, and `version: "pptxt-0.1"`.

## Stage D

Indexing

### Inputs
- Stage C aggregated JSONL:
  - Markdown chunks: `data/data_etl/md_data.jsonl`.
  - Text chunks: `data/data_etl/txt_data.jsonl`.
- Configured indexing paths (from `etl/config.py`):
  - `MD_BM25_INDEX_DIR`, `MD_QDRANT_INDEX_DIR`.
  - `TXT_BM25_INDEX_DIR`, `TXT_QDRANT_INDEX_DIR`.
- Embedding configuration (from `etl/config.py`):
  - Markdown: `MD_EMBEDDING_MODEL`, `MD_EMBEDDING_DIM`, `MD_QDRANT_BATCH_SIZE`, `MD_QDRANT_COLLECTION`.
  - Text: `TXT_EMBEDDING_MODEL`, `TXT_EMBEDDING_DIM`, `TXT_QDRANT_BATCH_SIZE`, `TXT_QDRANT_COLLECTION`.
- External libraries:
  - Whoosh for BM25 indexing.
  - Qdrant local storage + `sentence-transformers` for vector indexing.

### Outputs
- Whoosh (BM25) indexes:
  - Markdown: `data/data_index/md_bm25_storage/`.
  - Text: `data/data_index/txt_bm25_storage/`.
- Qdrant local vector stores:
  - Markdown: `data/data_index/md_qrant_storage/`.
  - Text: `data/data_index/txt_qrant_storage/`.
- Stage D metadata is merged into `data/extract_details.json` by `run_etl.py` (no standalone Stage D JSON).

### Norms
- BM25 indexing (Whoosh):
  - Markdown schema fields: `chunk_id`, `text`, `section`, `labels`, `paper_id`, `year` (defaults to 0 if absent).
  - Text schema fields: `chunk_id`, `text`, `paper_id`, `section`, `section_path`, `chunk_type`, `labels`, `has_math_loss`, `arxiv_ids`, `emails`, `urls`.
  - Skips empty-text records; records decode errors and blank lines.
- Qdrant indexing:
  - Embeddings generated with `SentenceTransformer(EMBEDDING_MODEL)`; dimension mismatch raises `ValueError`.
  - Collections are recreated by default (existing collection deleted if present).
  - Vectors use cosine distance and optional normalization; full record payloads are stored.
  - Skips empty-text records; batches upserts with configured batch sizes.
