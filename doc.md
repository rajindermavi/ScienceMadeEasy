
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
