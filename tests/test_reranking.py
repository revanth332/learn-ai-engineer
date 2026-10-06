import pytest
from src.retrieval.reranking import (
    CandidateDocument,
    RankedDocument,
    compute_cross_score,
    rerank,
    tokenize,
)


def test_compute_cross_score_exact_match():
    query = "deep learning python"
    doc = "This guide covers deep learning python with examples"
    score = compute_cross_score(query, doc)
    # Coverage = 1.0 (all 3 terms)
    # N-gram: "deep learning" and "learning python" both present -> 1.0
    # Proximity: indices 3, 4, 5 -> span = 5 - 3 + 1 = 3 -> 3 / 3 = 1.0
    # Total = 0.5*1.0 + 0.3*1.0 + 0.2*1.0 = 1.0
    assert pytest.approx(score, rel=1e-3) == 1.0


def test_compute_cross_score_no_match():
    query = "quantum computing"
    doc = "baking sourdough bread at home"
    assert compute_cross_score(query, doc) == 0.0


def test_compute_cross_score_partial_and_dispersed():
    query = "machine learning"
    # Doc mentions machine at start, learning far away at end
    doc = "machine tools are useful for many tasks including learning"
    score = compute_cross_score(query, doc)
    # Coverage = 1.0
    # N-gram = 0.0 ("machine learning" consecutive bigram not present)
    # Proximity: span = 8 - 0 + 1 = 9 -> 2 / 9 = ~0.222
    # Score = 0.5 * 1.0 + 0.3 * 0.0 + 0.2 * (2/9) = 0.5 + 0.0444 = 0.5444
    assert 0.5 < score < 0.6


def test_compute_cross_score_single_term():
    query = "python"
    doc = "modern python programming"
    score = compute_cross_score(query, doc)
    # Coverage = 1.0
    # N-gram = 1.0 (single term present)
    # Proximity = 1.0 (span = 1 -> 1/1 = 1.0)
    assert score == 1.0


def test_rerank_filtering_and_order():
    query = "machine learning models"
    candidates = [
        CandidateDocument(id="doc1", text="cooking pasta with cheese"),
        CandidateDocument(id="doc2", text="training machine learning models in python"),
        CandidateDocument(id="doc3", text="machine parts for manufacturing without learning"),
    ]

    ranked = rerank(query, candidates, top_n=2, min_score=0.1)
    assert len(ranked) == 2
    assert ranked[0].id == "doc2"
    assert ranked[1].id == "doc3"
    assert ranked[0].score > ranked[1].score
    assert all(isinstance(r, RankedDocument) for r in ranked)


def test_rerank_threshold():
    query = "neural networks"
    candidates = [
        CandidateDocument(id="doc1", text="introduction to neural networks"),  # high score
        CandidateDocument(id="doc2", text="neural connections in the brain"),   # partial match
    ]

    # Only return docs with score >= 0.8
    ranked = rerank(query, candidates, top_n=5, min_score=0.8)
    assert len(ranked) == 1
    assert ranked[0].id == "doc1"


def test_rerank_validation():
    with pytest.raises(ValueError, match="top_n"):
        rerank("test", [], top_n=0)

    with pytest.raises(ValueError, match="query"):
        rerank("   ", [], top_n=3)

    with pytest.raises(ValueError, match="min_score"):
        rerank("test", [], top_n=3, min_score=1.5)
