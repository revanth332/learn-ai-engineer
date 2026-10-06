import math
import pytest
from src.retrieval.bm25 import BM25Index, tokenize, BM25Result


def test_tokenize():
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]
    assert tokenize("   ") == []


def test_init_validation():
    with pytest.raises(ValueError, match="k1"):
        BM25Index(k1=-0.5)

    with pytest.raises(ValueError, match="b"):
        BM25Index(b=-0.1)

    with pytest.raises(ValueError, match="b"):
        BM25Index(b=1.2)


def test_add_and_properties():
    index = BM25Index()
    index.add("doc1", "The quick brown fox")
    index.add("doc2", "jumps over the lazy dog")

    assert index.corpus_size == 2
    assert index.doc_lengths["doc1"] == 4
    assert index.doc_lengths["doc2"] == 5
    assert index.avgdl == 4.5
    assert index.doc_freqs["the"] == 2
    assert index.doc_freqs["fox"] == 1


def test_add_validation():
    index = BM25Index()
    index.add("doc1", "hello world")

    with pytest.raises(ValueError, match="already exists"):
        index.add("doc1", "duplicate id")

    with pytest.raises(ValueError, match="empty"):
        index.add("doc2", "   !@#$   ")


def test_idf_calculation():
    index = BM25Index()
    index.add("doc1", "machine learning python")
    index.add("doc2", "deep learning python neural")
    index.add("doc3", "cooking food recipe")

    # 'python' appears in 2 of 3 docs
    # N=3, n=2: ln( (3 - 2 + 0.5) / (2 + 0.5) + 1 ) = ln( 1.5/2.5 + 1 ) = ln(1.6)
    expected_python_idf = math.log((3 - 2 + 0.5) / (2 + 0.5) + 1.0)
    assert pytest.approx(index.idf("python"), rel=1e-4) == expected_python_idf

    # 'neural' appears in 1 of 3 docs (rarer -> higher IDF)
    assert index.idf("neural") > index.idf("python")


def test_search_exact_keyword_ranking():
    index = BM25Index()
    index.add("doc1", "python language programming guide")
    index.add("doc2", "cooking italian pasta recipe")
    index.add("doc3", "python data science and machine learning with python")

    # Search for "python"
    # doc3 has higher term frequency (2 occurrences) -> higher score than doc1
    results = index.search("python", top_k=2)
    assert len(results) == 2
    assert results[0].doc_id == "doc3"
    assert results[1].doc_id == "doc1"
    assert results[0].score > results[1].score


def test_search_unmatched_query():
    index = BM25Index()
    index.add("doc1", "apple orange banana")
    results = index.search("helicopter", top_k=3)
    assert results == []


def test_search_validation():
    index = BM25Index()
    index.add("doc1", "apple orange")

    with pytest.raises(ValueError, match="top_k"):
        index.search("apple", top_k=0)

    with pytest.raises(ValueError, match="tokens"):
        index.search("   !@#$  ", top_k=2)
