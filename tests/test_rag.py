import pytest

from query.rag import is_mathy

@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Explain \\lambda notation in detail", True),
        ("How does an eigenvalue relate to stability?", True),
        ("Summarize spectral theory basics", True),
        ("Describe classical mechanics concepts", False),
        ("What is the role of probability in physics?", False),
    ],
)
def test_is_mathy_detects_mathy_queries(query, expected):
    assert is_mathy(query) is expected
