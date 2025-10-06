from query.md_query import hybrid_md_search_from_disk
import config

# Embedded Qdrant (folder-based)
resp = hybrid_md_search_from_disk(
    query="Cheeger inequality spectral gap for graph Laplacian",
    bm_index_path=config.MD_BM25_INDEX_DIR,
    qdrant_index_path=config.MD_QDRANT_INDEX_DIR,
    collection_name=config.MD_QDRANT_COLLECTION,
    embedding_model=config.MD_EMBEDDING_MODEL,
    topk=config.MD_TOPK,
    return_payloads=True
)
print(resp["results"][0])
