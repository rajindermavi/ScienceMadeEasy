import pytest

from ..data.etl.etl_stage_a import get_semantic_scholar_data

@pytest.mark.parametrize(
    ("arxiv_id", "empty_cite","empty_ref"),
    [
        ('2407.00703', [], [])
    ]
)
def test_get_semantic_scholar_data(arxiv_id,empty_cite,empty_ref):
    cite, ref = get_semantic_scholar_data(arxiv_id)
    cite_type, ref_type = type(cite), type(ref)
    assert cite_type is type(empty_cite)
    assert ref_type is type(empty_ref)