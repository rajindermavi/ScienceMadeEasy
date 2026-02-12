# TODO

1. Fix execution blockers
   - `query/rag_agent.py`: fix string quoting in `ngbr_direction`.
   - `query/retrieval.py`: remove undefined `self.md_data`/`self.txt_data` usage or initialize them properly.
2. Make retrieval robust when Qdrant isn’t configured
   - `query/index_query.py`: support local/embedded Qdrant or allow dense search to be disabled with a clear fallback.
   - Validate config early and fail with explicit errors.
3. Add deterministic, offline tests
   - Mock arXiv/Semantic Scholar and LLM calls.
   - Add unit tests for `IndexRetrieval.search`, `rrf_fuse`, sufficiency loop, and session memory.
4. Add CI with lint + tests
   - `.github/workflows/ci.yml`: run `pytest` and `ruff` (or `flake8`) + `mypy` baseline.
5. Harden LLM calls
   - `query/llm.py`: add retries/backoff, timeouts, request IDs.
   - Redact or summarize logs to avoid leaking prompts/keys.
6. Separate config by environment
   - `example.env` + `etl/config.py`: validate required env vars at startup.
   - Document defaults and a `.env.local` pattern in `readme.md`.
7. Make ETL reproducible
   - Pin exact arXiv/Semantic Scholar query params.
   - Log data versioning and store dataset manifest with hashes in `data/`.
8. Add data lineage metadata
   - Include `paper_id`, `arxiv_id`, `version`, and chunk provenance (stage, line offsets, file path).
9. Improve retrieval quality metrics
   - Add automated metrics (Recall@k, MRR@k, nDCG) with bootstrap confidence intervals.
   - Run as a script, not only notebooks.
10. Add RAG evaluation fixtures
    - Add a small curated dataset with expected answers.
    - Test for citation correctness and grounding.
11. Optimize perf and caching
    - Cache Qdrant payload lookups.
    - Avoid per‑ID scrolls; prefetch in batches.
12. Make UX expectations explicit
    - Document model requirements, minimum hardware, expected latency, and cost per query.
    - Add “known limits.”
13. Add basic observability
    - Structured logs with request IDs.
    - Per‑stage timings in ETL + query.
14. Add a simple API interface
    - Minimal FastAPI endpoint for query + citations for integration testing.
15. Provide a “dry run” mode
    - `run_etl.py` / ETL stages: allow processing N papers and skipping downloads.
