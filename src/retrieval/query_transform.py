"""Query Transformation & Expansion Module.

This module provides deterministic query expansion, multi-query decomposition,
HyDE prompt generation, and query set fusion for high-recall retrieval.
"""

from __future__ import annotations
import re

def tokenize(text: str) -> list[str]:
    """Tokenize string into lowercase alphanumeric tokens."""
    return re.findall(r"\w+", text.lower())

def expand_query_synonyms(query: str, synonym_map: dict[str, list[str]]) -> str:
    """Expands query terms with domain synonyms.

    For each word in `query` (case-insensitive matching):
    If the lowercased word exists in `synonym_map`, append its unique synonyms
    to the query string in parentheses or space-separated without duplicate terms.

    Args:
        query: The raw search string.
        synonym_map: Dictionary mapping lowercase canonical terms to list of synonyms.

    Returns:
        The expanded query string containing original tokens plus new unique synonyms.
        Example:
            query = "fast database"
            synonym_map = {"fast": ["quick", "low-latency"]}
            result = "fast database quick low-latency"
    """
    words = list(dict.fromkeys(tokenize(query)))
    existing_tokens = set(tokenize(query))
    for word in words:
        if word in synonym_map:
            for syn in synonym_map[word]:
                if syn.lower() not in existing_tokens:
                    query += (" " + syn)
                    existing_tokens.add(syn.lower())
    return query

def decompose_query(query: str) -> list[str]:
    """Decomposes a compound or comparative query into distinct standalone sub-queries.

    Splits queries across common conjunctions and comparison delimiters:
    - Conjunctions: ' and ', ', and ', ' as well as ', '; '
    - Comparisons: ' vs ', ' vs. ', ' versus ', ' compared to '

    Rules:
    - Strips leading/trailing whitespace and punctuation from each sub-query.
    - Filters out empty strings.
    - If no splitting delimiters are present, returns a list containing only the original stripped query.
    - Preserves case in individual sub-queries.

    Args:
        query: The raw compound user query.

    Returns:
        A list of non-empty cleaned sub-queries.
    """
    if query.strip() == "":
        return []
    splitting_delims = (', and ',' and ', ' as well as ', '; ',' vs ', ' vs. ', ' versus ', ' compared to ')
    delim_str = r"|".join([re.escape(delim) for delim in splitting_delims])
    sub_queries = re.split(delim_str,query,flags=re.IGNORECASE)
    for i in range(len(sub_queries)):
        sub_queries[i] = sub_queries[i].strip(" \t\r\n,;.")
    results = [s for s in sub_queries if s]
    return results

def generate_hyde_prompt(query: str, domain: str = "general") -> str:
    """Constructs a deterministic HyDE (Hypothetical Document Embeddings) prompt.

    The prompt instructs an LLM to hallucinate a factual, informative document passage
    that directly answers the query, optimizing downstream embedding similarity.

    Format requirements:
    "Please write a short, highly technical passage that answers the following {domain} question.\n"
    "Question: {query}\n"
    "Passage:"

    Args:
        query: The raw user search query.
        domain: Domain context (e.g., 'technical', 'medical', 'general'). Defaults to 'general'.

    Returns:
        The formatted HyDE generation prompt string.
    """
    hyde_prompt = f"Please write a short, highly technical passage that answers the following {domain} question.\nQuestion: {query}\nPassage:"
    return hyde_prompt


def fuse_transformed_queries(
    original_query: str,
    expanded_query: str | None = None,
    sub_queries: list[str] | None = None,
) -> list[str]:
    """Fuses and deduplicates original, expanded, and sub-queries into a clean retrieval set.

    Rules:
    - Original query is always first if non-empty.
    - Expanded query is added next (if provided and different from original query).
    - Sub-queries are appended in order (excluding any that are identical to original or expanded).
    - Case-insensitive deduplication, preserving the first-seen casing.
    - Ignores empty or whitespace-only strings.

    Args:
        original_query: The base user query.
        expanded_query: Optional synonym-expanded query.
        sub_queries: Optional list of decomposed sub-queries.

    Returns:
        Ordered list of unique, non-empty query variations.
    """
    query_variations = []
    if original_query:
        query_variations.append(original_query)
    if expanded_query and original_query.lower() != expanded_query.lower():
        query_variations.append(expanded_query)
    query_variations += (sub_queries or [])

    seen = set()
    result = []
    for query in query_variations:
        cleaned = query.strip()
        if cleaned and cleaned.lower() not in seen:
            result.append(cleaned)
            seen.add(cleaned.lower())

    return result
