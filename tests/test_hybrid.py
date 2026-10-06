import pytest
from src.retrieval.hybrid import reciprocal_rank_fusion, HybridRetriever, HybridResult
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25 import BM25Index


def test_rrf_single_list():
    ranked_list = [["doc1", "doc2", "doc3"]]
    # doc1: rank 1 -> 1 / (60 + 1) = 1/61
    # doc2: rank 2 -> 1 / (60 + 2) = 1/62
    # doc3: rank 3 -> 1 / (60 + 3) = 1/63
    results = reciprocal_rank_fusion(ranked_list, k=60)
    assert len(results) == 3
    assert results[0] == ("doc1", pytest.approx(1.0 / 61.0))
    assert results[1] == ("doc2", pytest.approx(1.0 / 62.0))
    assert results[2] == ("doc3", pytest.approx(1.0 / 63.0))


def test_rrf_multiple_lists_fusion():
    # Dense:  doc1 (#1), doc2 (#2)
    # Sparse: doc2 (#1), doc3 (#2)
    dense_ranks = ["doc1", "doc2"]
    sparse_ranks = ["doc2", "doc3"]

    results = reciprocal_rank_fusion([dense_ranks, sparse_ranks], k=60)
    # Scores:
    # doc1: 1/61 = ~0.01639
    # doc2: 1/62 + 1/61 = ~0.03252 -> WINNER (present in both)
    # doc3: 1/62 = ~0.01613
    assert results[0][0] == "doc2"
    assert pytest.approx(results[0][1]) == (1.0 / 62.0 + 1.0 / 61.0)
    assert results[1][0] == "doc1"
    assert results[2][0] == "doc3"


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_invalid_k():
    with pytest.raises(ValueError, match="k"):
        reciprocal_rank_fusion([["doc1"]], k=0)

    with pytest.raises(ValueError, match="k"):
        reciprocal_rank_fusion([["doc1"]], k=-5)


def test_hybrid_retriever_end_to_end():
    vector_store = VectorStore(dimension=2)
    bm25_index = BM25Index()
    retriever = HybridRetriever(vector_store, bm25_index)

    # doc1: high dense similarity, no keyword match
    retriever.add("doc1", "apple fruit red", [1.0, 0.0], metadata={"topic": "fruits"})
    # doc2: good dense similarity AND exact keyword match
    retriever.add("doc2", "python programming language code", [0.9, 0.1], metadata={"topic": "coding"})
    # doc3: exact keyword match, poor dense similarity
    retriever.add("doc3", "advanced python python scripts", [0.0, 1.0], metadata={"topic": "scripts"})

    # Query: text="python code", vector=[1.0, 0.0]
    results = retriever.search(
        query_text="python code",
        query_vector=[1.0, 0.0],
        top_k=3,
        rrf_k=60,
    )

    assert len(results) == 3
    # doc2 should rank high because it appears in both dense and sparse top ranks
    doc_ids = [r.doc_id for r in results]
    assert "doc2" in doc_ids
    assert results[0].metadata["topic"] in ["fruits", "coding"]
    assert all(isinstance(r, HybridResult) for r in results)


def test_hybrid_retriever_validation():
    vector_store = VectorStore(dimension=2)
    bm25_index = BM25Index()
    retriever = HybridRetriever(vector_store, bm25_index)

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("test", [1.0, 0.0], top_k=0)

    with pytest.raises(ValueError, match="rrf_k"):
        retriever.search("test", [1.0, 0.0], top_k=2, rrf_k=-1)
