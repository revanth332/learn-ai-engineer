"""In-memory Vector Store with exact cosine nearest neighbor search."""

from dataclasses import dataclass, field
from typing import Any
from src.retrieval.vector_math import cosine_similarity


@dataclass
class VectorRecord:
    """A stored vector with unique identifier and associated metadata."""

    id: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Result of a vector similarity search."""

    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """Exact nearest-neighbor in-memory vector database."""

    def __init__(self, dimension: int | None = None) -> None:
        """Initialize the vector store.

        Args:
            dimension: Optional fixed vector dimension. If provided, all added
                       vectors must match this dimension.
        """
        self.dimension = dimension
        self._records: dict[str, VectorRecord] = {}

    def add(
        self,
        id: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert a vector record into the store.

        Args:
            id: Unique identifier for the record.
            vector: Embedding vector.
            metadata: Optional dictionary of arbitrary metadata.

        Raises:
            ValueError: If id already exists, vector is empty, or vector dimension mismatches.
        """
        if vector is None or vector == []:
            raise ValueError("vector cannot be empty")
        if self.dimension is None:
            self.dimension = len(vector)
        if len(vector) != self.dimension:
            raise ValueError("vector dimension mismatch")
        if id in self._records:
            raise ValueError(f"Record with id '{id}' already exists")
        vector_record = VectorRecord(id, vector, metadata if metadata is not None else {})
        self._records[id] = vector_record

    def get(self, id: str) -> VectorRecord | None:
        """Retrieve a record by its unique ID. Returns None if not found."""
        return self._records.get(id)

    def delete(self, id: str) -> bool:
        """Delete a record by ID. Returns True if deleted, False if not found."""
        if id in self._records:
            del self._records[id]
            return True
        return False

    def __len__(self) -> int:
        return len(self._records)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 3,
        threshold: float | None = None,
    ) -> list[SearchResult]:
        """Perform exact nearest neighbor search using cosine similarity.

        Args:
            query_vector: Embedding vector representing the query.
            top_k: Maximum number of top matches to return (must be > 0).
            threshold: Optional minimum cosine similarity threshold (e.g. 0.7).

        Returns:
            List of SearchResult objects sorted descending by score (highest similarity first).

        Raises:
            ValueError: If top_k <= 0, query_vector is empty, or dimension mismatches.
        """
        if query_vector is None or query_vector == []:
            raise ValueError("vector cannot be empty")
        if top_k is None or top_k <= 0:
            raise ValueError("top_k must be positive")
        if len(query_vector) != self.dimension:
            raise ValueError("query_vector dimension mismatch")

        results:list[SearchResult] = []
        for _,record in self._records.items():
            score = cosine_similarity(query_vector,record.vector)
            result = SearchResult(id=record.id,score=score,metadata=record.metadata)
            results.append(result)

        if threshold is not None:
            results = [result for result in results if result.score >= threshold]       
        results.sort(key=lambda x : -x.score)
        return results[:top_k]