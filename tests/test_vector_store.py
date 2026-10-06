import pytest
from src.retrieval.vector_store import VectorStore, VectorRecord, SearchResult


def test_add_and_get():
    store = VectorStore(dimension=3)
    store.add("doc1", [1.0, 0.0, 0.0], metadata={"text": "chunk 1"})
    
    record = store.get("doc1")
    assert record is not None
    assert record.id == "doc1"
    assert record.vector == [1.0, 0.0, 0.0]
    assert record.metadata["text"] == "chunk 1"
    assert len(store) == 1


def test_add_auto_sets_dimension_if_none():
    store = VectorStore()
    store.add("doc1", [1.0, 2.0, 3.0])
    assert store.dimension == 3


def test_add_dimension_mismatch_raises():
    store = VectorStore(dimension=3)
    with pytest.raises(ValueError, match="dimension"):
        store.add("doc1", [1.0, 2.0])


def test_add_duplicate_id_raises():
    store = VectorStore(dimension=2)
    store.add("doc1", [1.0, 0.0])
    with pytest.raises(ValueError, match="already exists"):
        store.add("doc1", [0.0, 1.0])


def test_delete_record():
    store = VectorStore(dimension=2)
    store.add("doc1", [1.0, 0.0])
    assert len(store) == 1
    assert store.delete("doc1") is True
    assert store.delete("doc1") is False
    assert len(store) == 0


def test_search_ranking_and_top_k():
    store = VectorStore(dimension=2)
    # Target query is [1.0, 0.0]
    store.add("exact", [1.0, 0.0], metadata={"name": "exact"})        # sim = 1.0
    store.add("close", [0.9, 0.1], metadata={"name": "close"})        # sim ~ 0.99
    store.add("orthogonal", [0.0, 1.0], metadata={"name": "ortho"})    # sim = 0.0
    store.add("opposite", [-1.0, 0.0], metadata={"name": "opp"})       # sim = -1.0

    results = store.search([1.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].id == "exact"
    assert pytest.approx(results[0].score, rel=1e-3) == 1.0
    assert results[1].id == "close"


def test_search_threshold_filtering():
    store = VectorStore(dimension=2)
    store.add("doc1", [1.0, 0.0])     # sim = 1.0
    store.add("doc2", [0.5, 0.5])     # sim ~ 0.707
    store.add("doc3", [0.0, 1.0])     # sim = 0.0

    # Only retrieve docs with similarity >= 0.8
    results = store.search([1.0, 0.0], top_k=5, threshold=0.8)
    assert len(results) == 1
    assert results[0].id == "doc1"


def test_search_empty_store():
    store = VectorStore(dimension=2)
    results = store.search([1.0, 0.0], top_k=3)
    assert results == []


def test_search_invalid_parameters():
    store = VectorStore(dimension=2)
    store.add("doc1", [1.0, 0.0])

    with pytest.raises(ValueError, match="top_k"):
        store.search([1.0, 0.0], top_k=0)

    with pytest.raises(ValueError, match="dimension"):
        store.search([1.0, 0.0, 0.0], top_k=3)
