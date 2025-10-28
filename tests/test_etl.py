import pytest

import arxiv

import config
import inspect
from logs.logger import get_logger
logger = get_logger(log_name='run_etl',log_path=config.DEFAULT_LOG_DIR/'test.log')

from data.etl.etl_stage_a import (
    build_arxiv_query,
    arxiv_extract,
    get_semantic_scholar_data,
    arxiv_client_search
)

def test_get_semantic_scholar_data(arxiv_id='2407.00703'):
    logger.info(f'Starting test: {inspect.currentframe()}')
    cite, ref = get_semantic_scholar_data(arxiv_id)
    cite_type, ref_type = type(cite), type(ref)
    assert cite_type is type([])
    assert ref_type is type([])

def test_get_arxiv_data():
    logger.info(f'Starting test: {inspect.currentframe()}')
    arxiv_query = build_arxiv_query(['almost Mathieu operator'],['math-ph'])
    logger.info(f'Query created:\n {arxiv_query}')
    collection = arxiv_client_search(arxiv_query,300)
    assert len(collection) > 50