"""Cross-Encoder Re-ranking from first principles for two-stage retrieval."""

from dataclasses import dataclass, field
from typing import Any
import re


@dataclass
class CandidateDocument:
    """Document candidate retrieved from Stage-1 retrieval."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RankedDocument:
    """Document scored and ranked by Stage-2 Cross-Encoder."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    """Tokenize string into lowercase alphanumeric tokens."""
    return re.findall(r"\w+", text.lower())


def compute_cross_score(query: str, doc_text: str) -> float:
    """Compute fine-grained cross-attention lexical-semantic score between query and document.

    Scoring Components (bounded [0.0, 1.0]):
    1. Coverage (50%): Ratio of unique query terms present in document.
    2. Contiguous N-Gram Match (30%): Fraction of consecutive query bigrams present in doc.
       (If query has only 1 word, ngram_score = 1.0 if word present else 0.0).
    3. Proximity Density (20%): How closely packed the matching query terms appear in doc.
       Span = (max_index - min_index + 1) of first occurrences of matching terms.
       Proximity = len(matching_terms) / Span. If no matches, proximity = 0.0.

    Total Score:
        If coverage == 0.0 -> 0.0
        Else -> round(0.5 * coverage + 0.3 * ngram_score + 0.2 * proximity, 4)

    Args:
        query: Query string.
        doc_text: Document text string.

    Returns:
        Score float between 0.0 and 1.0.
    """
    query_terms = tokenize(query)
    doc_text_arr = tokenize(doc_text)
    n = len(query_terms)

    # coverage
    matched_terms = []
    uniq_terms = set(query_terms)
    for term in uniq_terms:
        if term in doc_text_arr:
            matched_terms.append(term)
    if len(matched_terms) == 0:
        return 0.0
    
    coverage = len(matched_terms) / len(uniq_terms)

    # N Gram Match
    ngram_score = 0.0
    if n == 1:
        if query_terms[0] in doc_text_arr:
            ngram_score = 1.0
    else:
        pair_count = 0
        for i in range(n-1):
            for j in range(len(doc_text_arr)-1):
                if query_terms[i:i+2] == doc_text_arr[j:j+2]:
                    pair_count += 1
                    break
        ngram_score = pair_count / (n-1)
    
    # proximity density
    min_index = float("inf")
    max_index = -1
    for term in matched_terms:
        min_index = min(min_index,doc_text_arr.index(term))
        max_index = max(max_index,doc_text_arr.index(term))
    span = max_index - min_index + 1
    proximity_score = len(matched_terms)/span

    return round(0.5 * coverage + 0.3 * ngram_score + 0.2 * proximity_score,4)

def rerank(
    query: str,
    candidates: list[CandidateDocument],
    top_n: int = 3,
    min_score: float = 0.0,
) -> list[RankedDocument]:
    """Re-rank candidate documents using cross-encoder scoring.

    Args:
        query: User search query.
        candidates: List of CandidateDocument objects from Stage 1.
        top_n: Number of top re-ranked candidates to return (> 0).
        min_score: Minimum cross-score threshold (0.0 to 1.0).

    Returns:
        List of RankedDocument objects sorted descending by score.

    Raises:
        ValueError: If top_n <= 0 or query is empty or not (0.0 <= min_score <= 1.0).
    """
    query = query.strip()
    if query == "":
        raise ValueError("query is empty")
    if top_n <= 0:
        raise ValueError("top_n should be positive")
    if min_score < 0 or min_score > 1:
        raise ValueError("invalid min_score")

    ranked_documents = []
    for candidate in candidates:
        cross_score = compute_cross_score(query,candidate.text)
        if cross_score >= min_score:
            ranked_documents.append(RankedDocument(candidate.id,candidate.text,cross_score,candidate.metadata))
    
    ranked_documents.sort(key=lambda x : -x.score)
    return ranked_documents[:top_n]