# Science Made Easy

## Summary
A math aware Retrieval-Augmented Generation protoype for mathematical research papers. 
The prototype is shipped with an indexed set of papers focused generally on Quasiperiodic Schrodinger Operators and the Almost Mathieu Operator in particular.

The papers are sourced from Arxiv.org. Most all mathematical papers are posted with LaTex source, so that source is used as our text corpus.
Latex is transformed into md and stripped txt format and subsequently chunked for later retrieval.
Indexing and retrival uses BM25 and Qdrant, each the txt and md sources are combine retrieval methods using reciproval rank fusion.
Finally the txt and md sources combined and reranked using a cross encoder.

## Quick start

```
git clone <repo-url>
cd project
pip install -r requirements.txt
```

Rename example.env to .env and update OPENAI_API_KEY

```
streamlit run app.py
```

## Overview

```
SME/
├── data/                     
│   ├── data_etl/               # Raw data, chunk JSON, JSONL storage
│   ├── data_index/             # Index Storage
│   └── etl/
│       ├── etl_stage_a.py
│       ├── etl_stage_b.py
│       ├── etl_stage_c_md.py
│       ├── etl_stage_c_txt.py
│       ├── etl_stage_d_md.py
│       ├── etl_stage_d_txt.py
│       └── models.py
├── logging/
│   ├── etl.log
│   └── query.log
├── query/
│   ├── index_query.py
│   └── rag.py
├── app.py
├── config.py
├── requirments.txt
├── run_etl.py                  # Rerun to retrieve a new corpus
├── run_query.py                # Run query outside runtime frontend
└── README.md
```

The ```config.py``` file contains settings for the indexing steps - embedding dimensions, models etc.

## Running the ETL

In order to rerun the ETL you will have to install Latexpand, Pandoc, and Detex.
Latexpand and Detex are included in texlive-extra-utils.

```
sudo apt update

sudo apt install pandoc texlive-extra-utils
```