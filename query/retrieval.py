import json

from query.index_query import hybrid_search_from_disk, rerank
import etl.config as config
from etl.config import MD_JSON, TXT_JSON,EXTRACT_DETAILS


class IndexRetrieval:
    def __init__(self):
        self.md_data = None
        self.txt_data = None
        self.papers = None
        self.load_metadata()

    def _load_metadata(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_metadata(self):
        self.md_data = self._load_metadata(str(MD_JSON))
        self.txt_data = self._load_metadata(str(TXT_JSON))
        extract_details = self._load_metadata(str(EXTRACT_DETAILS))
        self.papers = extract_details.get('papers')

    def load_metadata_for_result(self, result):
        if self.md_data is None or self.txt_data is None:
            self.load_metadata()
        if not isinstance(result, dict):
            return None, None
        chunk_id = result.get("chunk_id") or result.get("id")
        if not chunk_id:
            return None, None
        if chunk_id in self.md_data:
            return chunk_id, self.md_data[chunk_id]
        if chunk_id in self.txt_data:
            return chunk_id, self.txt_data[chunk_id]
        return None, None

    def metadata_retrieval(self, results):
        metadata = {}
        for result in results:
            chunk_id, data = self.load_metadata_for_result(result)
            if chunk_id:
                metadata[chunk_id] = data
        return metadata

    def neighbors_metadata(self, results):
        metadata = {}
        for result in results:
            chunk_id, data = self.load_metadata_for_result(result)
            if chunk_id:
                metadata[chunk_id] = data
        return metadata

    @staticmethod
    def query_retrieval(query,md_topk = config.MD_TOPK,txt_topk = config.TXT_TOPK):
        md_use_server = bool(config.MD_QDRANT_HOST)
        resp_md = hybrid_search_from_disk(
            query=query,
            bm_index_path=config.MD_BM25_INDEX_DIR,
            qdrant_index_path=None if md_use_server else config.MD_QDRANT_INDEX_DIR,
            qdrant_host=config.MD_QDRANT_HOST if md_use_server else None,
            qdrant_port=config.MD_QDRANT_PORT if md_use_server else None,
            qdrant_api_key=config.MD_QDRANT_API_KEY if md_use_server else None,
            collection_name=config.MD_QDRANT_COLLECTION,
            embedding_model=config.MD_EMBEDDING_MODEL,
            topk=md_topk,
            source="md",
            return_payloads=True,
        )
        txt_use_server = bool(config.TXT_QDRANT_HOST)
        resp_txt = hybrid_search_from_disk(
            query=query,
            bm_index_path=config.TXT_BM25_INDEX_DIR,
            qdrant_index_path=None if txt_use_server else config.TXT_QDRANT_INDEX_DIR,
            qdrant_host=config.TXT_QDRANT_HOST if txt_use_server else None,
            qdrant_port=config.TXT_QDRANT_PORT if txt_use_server else None,
            qdrant_api_key=config.TXT_QDRANT_API_KEY if txt_use_server else None,
            collection_name=config.TXT_QDRANT_COLLECTION,
            embedding_model=config.TXT_EMBEDDING_MODEL,
            topk=txt_topk,
            source="txt",
            return_payloads=True,
        )

        results = [*resp_md["results"], *resp_txt["results"]]
        rerank_results = rerank(query, results)
        return rerank_results
    
    def search(self,query,k=10):
        return self.query_retrieval(query,md_topk=k,txt_topk=k)
