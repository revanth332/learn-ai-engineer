import pytest
from src.retrieval.chunking import chunk_text


def test_chunk_text_empty():
    assert chunk_text("", chunk_size=10, chunk_overlap=0) == []


def test_chunk_text_shorter_than_chunk_size():
    assert chunk_text("hello", chunk_size=10, chunk_overlap=2) == ["hello"]


def test_chunk_text_exact_match():
    assert chunk_text("abcdef", chunk_size=6, chunk_overlap=0) == ["abcdef"]


def test_chunk_text_no_overlap():
    text = "abcdefghij"  # 10 chars
    # chunk_size 4, overlap 0 -> "abcd", "efgh", "ij"
    chunks = chunk_text(text, chunk_size=4, chunk_overlap=0)
    assert chunks == ["abcd", "efgh", "ij"]


def test_chunk_text_with_overlap():
    text = "abcdefghij"
    # chunk_size 4, overlap 2 -> step = 2
    # 0..4: "abcd"
    # 2..6: "cdef"
    # 4..8: "efgh"
    # 6..10: "ghij"
    chunks = chunk_text(text, chunk_size=4, chunk_overlap=2)
    assert chunks == ["abcd", "cdef", "efgh", "ghij"]


def test_chunk_text_tail_handling():
    text = "1234567"
    # chunk_size 4, overlap 1 -> step = 3
    # 0..4: "1234"
    # 3..7: "4567"
    chunks = chunk_text(text, chunk_size=4, chunk_overlap=1)
    assert chunks == ["1234", "4567"]


def test_chunk_text_invalid_parameters():
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text("test", chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text("test", chunk_size=-5, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text("test", chunk_size=10, chunk_overlap=-1)

    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text("test", chunk_size=10, chunk_overlap=10)

    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text("test", chunk_size=10, chunk_overlap=12)
