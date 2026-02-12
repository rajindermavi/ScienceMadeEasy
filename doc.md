
## Overview

```
science_made_easy/
├── analysis/                       # Analysis notebooks for performance evaluation
│   ├── demo_rag.ipynb
│   ├── evaluate_rag.ipynb
│   ├── evaluate_retrieval.ipynb
│   ├── EvaluationAvgs.png
│   ├── generated_queries.jsonl
│   ├── notebook_bootstrap.py
│   └── utilities.py
├── data/
│   ├── data_etl/                   # Raw data, chunk JSON, JSONL storage
│   ├── data_index/                 # bm25 storage
│   └── extract_details.json
├── docker/                         # Qdrant server docker files
│   ├── storage                     # Qdrant storage
│   ├── docker-compose.yml
│   └── Makefile
├── etl/
│   ├── config.py
│   ├── indexing_utils.py
│   ├── models.py
│   ├── stg_a_extract.py            # Extract files from arXiv.org
│   ├── stg_b_conversion.py         # Convert latex folders to single md and txt files
│   ├── stg_c_md_chunking.py        # chunking
│   ├── stg_c_txt_chunking.py       #   ''  
│   ├── stg_d_md_indexing.py        # Index with bm25 and Qdrant
│   └── stg_d_txt_indexing.py       #   ''  ''  ''  ''  ''  ''
├── log/
│   └── logger.py
├── query/
│   ├── index_query.py              # Match query angainst bm25 and qdrant indexes
│   ├── llm.py                      # Owns llm calls
│   ├── nlp.py                      # Simple 'old fashioned' nlp 
│   ├── prompt.py                   # Prompt constructions
│   ├── rag_agent.py                # Owns RAG agent
│   └── retrieval.py                # Fetches stored text and metadata
├── tests/
│   ├── conftest.py
│   ├── test_etl.py
│   └── test_rag.py
├── app.py                          # Streamlit app - main entry
├── doc.md
├── example.env
├── readme.md
├── requirements-dev.txt
├── requirements.txt
└── run_etl.py                      # Rerun to retrieve a new corpus
```

The `etl/config.py` file contains settings for the indexing steps - embedding dimensions, models, etc.
