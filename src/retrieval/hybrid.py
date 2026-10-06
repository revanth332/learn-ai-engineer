"""Hybrid search combining dense vector search and sparse BM25 with Reciprocal Rank Fusion (RRF)."""

from typing import Any
from dataclasses import dataclass, field
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25 import BM25Index


@dataclass
class HybridResult:
    """Result of a fused hybrid search."""

    doc_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists of document IDs using Reciprocal Rank Fusion (RRF).

    Formula for document d across ranked lists:
        RRF_Score(d) = sum( 1.0 / (k + rank_m(d)) ) for all lists m containing d
        where rank_m(d) is 1-based (i.e. first place is rank=1, second is rank=2).

    Args:
        ranked_lists: List of ranked doc_id lists from different retrievers.
        k: Smoothing constant parameter (must be > 0, default=60).

    Returns:
        List of (doc_id, fused_score) tuples sorted descending by fused_score.

    Raises:
        ValueError: If k <= 0.
    """
    if k <= 0:
        raise ValueError("k value must be positive")
    rrf_scores = {}
    result = []
    for d_list in ranked_lists:
        for index,d in enumerate(d_list):
            d_score = (1.0/(k+index+1))
            rrf_scores[d] = rrf_scores.get(d,0) + d_score
    for key,value in rrf_scores.items():
        result.append((key,value))
    result.sort(key=lambda x: -x[1])
    return result

class HybridRetriever:
    """Dual-engine hybrid retriever combining Dense Vectors and Sparse BM25."""

    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index) -> None:
        """Initialize hybrid retriever with vector and sparse engines."""
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def add(
        self,
        doc_id: str,
        text: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Index a document into both vector store and BM25 index.

        Args:
            doc_id: Unique document identifier.
            text: Document text for BM25 indexing.
            vector: Embedding vector for dense similarity search.
            metadata: Optional dictionary of document metadata.
        """
        self.bm25_index.add(doc_id,text)
        self.vector_store.add(doc_id,vector,metadata)

    def search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 3,
        rrf_k: int = 60,
    ) -> list[HybridResult]:
        """Perform hybrid retrieval using RRF fusion.

        1. Perform dense vector search on vector_store (fetch top_k * 2 candidates).
        2. Perform sparse keyword search on bm25_index (fetch top_k * 2 candidates).
        3. Fuse candidate rankings using reciprocal_rank_fusion.
        4. Return top_k HybridResult objects with metadata attached.

        Args:
            query_text: Search query string for lexical search.
            query_vector: Search query embedding for semantic search.
            top_k: Number of final results to return (must be > 0).
            rrf_k: Smoothing constant for RRF (must be > 0).

        Returns:
            List of HybridResult objects sorted descending by fused RRF score.

        Raises:
            ValueError: If top_k <= 0 or rrf_k <= 0.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        vector_result = self.vector_store.search(query_vector,top_k*2)
        try:
            bm25_result = self.bm25_index.search(query_text,top_k*2)
        except:
            bm25_result = []

        ranked_lists = [[item.id for item in vector_result],[item.doc_id for item in bm25_result]]
        fused_ranks = reciprocal_rank_fusion(ranked_lists,rrf_k)
        final_results = []
        for doc_id,score in fused_ranks[:top_k]:
            record = self.vector_store.get(doc_id)
            metadata = record.metadata if record else {}
            h_result = HybridResult(doc_id,score,metadata)
            final_results.append(h_result)
        return final_results