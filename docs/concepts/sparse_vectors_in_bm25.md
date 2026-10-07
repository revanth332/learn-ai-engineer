# Sparse Vectors and Vocabulary Representation in BM25

This document explains how **sparse vectors** and **vocabulary spaces** are mapped from theoretical mathematics to practical Python data structures in our BM25 implementation (`src/retrieval/bm25.py`).

---

## 1. Theoretical Concepts vs. Python Implementation

### A. The Vocabulary Space ($V$)
* **Theory:** In information retrieval, the vocabulary $V = \{w_1, w_2, \dots, w_{|V|}\}$ is the set of all unique words observed across the entire document corpus.
* **In `bm25.py`:**
  ```python
  self.doc_freqs: dict[str, int] = {}
  ```
  * The keys of `self.doc_freqs` represent the complete vocabulary set $V$.
  * The total vocabulary size $|V|$ is simply `len(self.doc_freqs)`.

---

### B. Dense Array vs. Sparse Dictionary Representation
* **Theoretical Dense Representation:**
  If the corpus has 50,000 unique words, a standard dense vector for a document containing only *"python fast"* would look like a 50,000-element array:
  $$\vec{d} = [0, 0, 0, \dots, 3 (\text{'python'}), \dots, 1 (\text{'fast'}), \dots, 0]$$
  * Storing 49,998 zeros in memory per document is computationally wasteful ($O(|V|)$ memory complexity).

* **Sparse Dictionary Representation in `bm25.py`:**
  ```python
  self.doc_term_freqs[doc_id] = {"python": 3, "fast": 1}
  ```
  * Only terms with non-zero frequencies ($\text{tf} > 0$) are explicitly stored ($O(L)$ memory complexity, where $L \ll |V|$ is document length).
  * Any unobserved term inherently has a value of `0`.

---

### C. The Sparse Vector Query ($0$-Handling)
During the search phase, querying for term weights reflects the implicit zeros of the sparse vector:

```python
# From BM25Index.search()
tf = self.doc_term_freqs[doc_id].get(term, 0)
```

* If `term` is in `self.doc_term_freqs[doc_id]`, its non-zero weight is retrieved.
* If `term` is not present, `.get(term, 0)` evaluates to `0`, accurately mirroring the zero-dimension in the sparse vector.

---

## 2. Summary Mapping Table

| Mathematical Concept | Theoretical Form | Implementation in `bm25.py` | Line / Attribute |
| :--- | :--- | :--- | :--- |
| **Global Vocabulary** | $V = \{w_1, \dots, w_N\}$ | `set(self.doc_freqs.keys())` | `self.doc_freqs` |
| **Vocabulary Size** | $|V|$ | `len(self.doc_freqs)` | `len(self.doc_freqs)` |
| **Sparse Document Vector** | $\vec{d} \in \mathbb{R}^{|V|}$ (mostly 0s) | `dict[str, int]` (only non-zeros) | `self.doc_term_freqs[doc_id]` |
| **Zero Dimension Lookup** | $d_i = 0$ | `.get(term, 0)` | `search()` method |
