# Query Transformation & Expansion

Query Transformation is the stage in a RAG pipeline that reformulates raw user input before retrieval to solve the vocabulary mismatch and ambiguity problem.

---

## 1. Core Terminology

- **Vocabulary Mismatch Problem:** The user phrases a query using words different from the indexed documents (e.g., query: *"car stalled"*, doc: *"engine failure and battery outage"*).
- **Query Expansion:** Adding synonyms, related technical terms, and alternative phrasings to a query to increase sparse/dense recall.
- **Multi-Query Decomposition:** Splitting a complex, multi-hop question into independent sub-queries to retrieve diverse pieces of evidence.
- **HyDE (Hypothetical Document Embeddings):** Generating a plausible, synthetic answer/document with an LLM first, then embedding the *hypothetical document* rather than the short raw query.
- **Query Rewriting / Disambiguation:** Resolving coreferences (e.g. converting *"How much does it cost?"* to *"How much does Stripe subscription cost?"* using chat history).

---

## 2. Why Query Transformation Matters

Raw user queries are typically:
1. **Too short** (1–4 words) $\rightarrow$ Low semantic signal in embedding space.
2. **Ambiguous or full of slang** $\rightarrow$ Poor BM25 keyword matching.
3. **Compound questions** $\rightarrow$ A single vector cannot match two distinct document spaces simultaneously.

### The 3 Core Strategies

| Strategy | When to Use | Mechanism |
| :--- | :--- | :--- |
| **Synonym & Term Expansion** | Domain-specific search, keyword mismatch | Augments query with domain synonyms and weighted variants. |
| **Multi-Query Decomposition** | Multi-hop / comparative queries (e.g. *"Compare Postgres vs Redis latency"*) | Breaks query into $N$ distinct sub-queries executed in parallel. |
| **Hypothetical Document Generation (HyDE)** | Short, conceptual, or question-style queries | LLM hallucinates a dummy answer; the dummy answer's embedding is closer to real documents than the question's embedding. |

---

## 3. Real-World Engineering Example

### Raw User Query:
> *"Why did my payment get blocked?"*

1. **HyDE Synthetic Passage:**
   > *"Payment processing failure code 402 or 403 occurs when the card issuer detects suspected fraud, insufficient funds, or automated velocity limits on billing addresses."*
2. **Sparse/Dense Retrieval:**
   - Embeds the synthetic passage $\rightarrow$ Matches internal support doc: *"Resolving 402/403 Transaction Declines and Velocity Rule Blocks"*.
3. **Result:**
   - The user gets the exact technical doc even though they never mentioned *"402"*, *"declines"*, or *"velocity limits"*.

---

## 4. Architecture & Step-by-Step Procedure

```
Raw Query
   │
   ├── [1. Query Expansion] ──────> Synonyms / Disambiguated Query
   ├── [2. Decomposition]    ──────> Sub-queries [Q1, Q2, ...]
   └── [3. HyDE Generator]   ──────> Synthetic Passage
```

### What You Will Build in `src/retrieval/query_transform.py`:
1. **`expand_query_synonyms(query, synonym_map)`**: Replaces or augments tokens with known domain synonyms without duplicating terms.
2. **`decompose_query(query)`**: Deterministically splits compound conjunctions (`" and "`, `", and "`, `";"`, `" vs "`, `" compared to "`) into standalone atomic sub-queries.
3. **`generate_hyde_prompt(query)`**: Formats an exact zero-shot hallucination prompt template for an LLM to generate a hypothetical document passage.
4. **`fuse_transformed_queries(original_query, expanded, sub_queries)`**: Deduplicates and unifies all transformed query variations into an optimal query set for downstream retrieval.
