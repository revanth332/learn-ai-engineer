"""Okapi BM25 sparse keyword retrieval from first principles."""

import math
import re
from dataclasses import dataclass


@dataclass
class BM25Result:
    """Result of a BM25 sparse retrieval search."""

    doc_id: str
    score: float


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """Okapi BM25 inverted index for lexical keyword search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize BM25 index with tuning parameters.

        Args:
            k1: Term frequency saturation parameter (typically 1.2 to 2.0).
            b: Document length normalization parameter (0.0 to 1.0).

        Raises:
            ValueError: If k1 < 0 or not (0.0 <= b <= 1.0).
        """
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not (0.0 <= b <= 1.0):
            raise ValueError("b must be between 0.0 and 1.0")

        self.k1 = k1
        self.b = b
        self.doc_lengths: dict[str, int] = {}
        self.doc_term_freqs: dict[str, dict[str, int]] = {}
        self.doc_freqs: dict[str, int] = {}  # term -> number of docs containing term

    @property
    def corpus_size(self) -> int:
        """Total number of documents indexed."""
        return len(self.doc_lengths)

    @property
    def avgdl(self) -> float:
        """Average document length across the indexed corpus."""
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def add(self, doc_id: str, text: str) -> None:
        """Add a document to the index.

        Args:
            doc_id: Unique identifier for the document.
            text: Document text content.

        Raises:
            ValueError: If doc_id already exists or tokenized text is empty.
        """
        if doc_id in self.doc_lengths:
            raise ValueError("doc_id already exists")
        
        tokens = tokenize(text)
        if not tokens:
            raise ValueError("empty text!")

        self.doc_lengths[doc_id] = len(tokens)
        self.doc_term_freqs[doc_id] = {}
        for token_id in tokens:
            self.doc_term_freqs[doc_id][token_id] = self.doc_term_freqs[doc_id].get(token_id,0) + 1
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self.doc_freqs[token] = self.doc_freqs.get(token,0) + 1

    def idf(self, term: str) -> float:
        """Compute Robertson-Spärck Jones IDF with Lucene floor (+1).
        n(t) -> number of documents containing term t
        N -> number of documents
        0.5 -> RSJ Smoothing constant
        +1.0 -> Lucene's floor adjustment( make sure the idf result value strictly greater than 0)

        Formula:
            ln( (N - n(t) + 0.5) / (n(t) + 0.5) + 1.0 )

        Args:
            term: Lowercase query term.

        Returns:
            IDF weight as float. Returns 0.0 if index is empty.
        """
        N = len(self.doc_lengths)
        if N == 0:
            return 0.0
        term = term.lower()
        nt = self.doc_freqs.get(term,0)
        return math.log(1+((N-nt+0.5)/(nt+0.5)))
        

    def search(self, query: str, top_k: int = 3) -> list[BM25Result]:
        """Score and rank indexed documents for a search query.

        BM25 Formula for term t in doc D:
            IDF(t) * (tf(t, D) * (k1 + 1)) / (tf(t, D) + k1 * (1 - b + b * (|D| / avgdl)))

        Args:
            query: Raw query string.
            top_k: Number of highest-ranking documents to return (must be > 0).

        Returns:
            List of BM25Result objects sorted descending by score. Only includes docs with score > 0.

        Raises:
            ValueError: If top_k <= 0 or query has no valid tokens.
        """
        if top_k <= 0:
            raise ValueError("top_k should be positive")
        tokens = tokenize(query)
        if len(tokens) == 0:
            raise ValueError("query has no valid tokens")
        scores = []
        for doc_id in self.doc_lengths:
            doc_score = 0
            for term in tokens:
                tf = self.doc_term_freqs[doc_id].get(term,0)
                D = self.doc_lengths[doc_id]
                term_score = self.idf(term) * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (D / self.avgdl)))
                doc_score += term_score
            if doc_score > 0:
                result = BM25Result(doc_id,doc_score)
                scores.append(result)

        scores.sort(key=lambda x: -x.score)
        return scores[:top_k]