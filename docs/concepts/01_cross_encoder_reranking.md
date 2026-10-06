# Module 01: Cross-Encoder Re-Ranking

## 1. What Are We Trying to Achieve?
In a search engine or RAG system, our first search (Stage 1: Vector Search + BM25) fetches 50 candidate documents very quickly. However, some of those 50 documents are **false alarms**—they contain the search words, but in the wrong order or spread too far apart.

Our goal in **Stage 2 (Re-Ranking)** is to take those candidate documents and score them with high precision, so the best 3 documents end up at the very top before we hand them to the LLM.

---

## 2. Plain-English Term Definitions

| Jargon Term | Plain-English Meaning | Real-World Analogy |
|---|---|---|
| **Lexical** | Exact word matches and spelling. | Searching with `Ctrl + F` for the exact word `"python"`. |
| **Semantic** | The underlying meaning or concept, regardless of exact words. | Understanding that `"automobile"` means `"car"`. |
| **Cross-Attention** | Analyzing the **Query** and the **Document** *together at the same time* to see how words relate to each other. | Reading a question and an answer side-by-side to see if they actually fit together. |
| **Bi-Encoder** | Encoding Query and Document *separately* into vectors, then comparing the two vectors. | Reading the question on Monday, reading the answer on Friday, and guessing if they match based on memory. |
| **Fine-Grained** | Checking small, important details (word order, distance between words, exact phrasing) rather than just high-level summary. | Checking not just if a pizza has cheese, but whether the cheese is melted on the crust or still in the wrapper. |
| **N-gram (Bigram)** | A pair of two consecutive words that appear side-by-side. | In `"deep learning"`, the pair `("deep", "learning")` is a bigram. |
| **Proximity Density** | How close together the search words appear inside the document. | If you search `"credit card"`, are the words `"credit"` and `"card"` right next to each other, or 200 words apart? |

---

## 3. Why We Need This (The Problem We Are Solving)

Imagine you search: **`"machine learning in python"`**

- **Document A:** `"I bought a washing machine yesterday, and I am learning python today."`
- **Document B:** `"Here is a complete guide to machine learning in python with code."`

Notice that **Document A** contains the words `"machine"`, `"learning"`, and `"python"`. A basic word search or simple vector search might think Document A is a great match!

A **Cross-Encoder Re-ranker** checks:
1. **Are the words right next to each other?** (Document B has `"machine learning"` together; Document A has them 7 words apart).
2. **Are they in the right order?**
3. **How much of the query is covered?**

Result: Document B gets a score of `1.0`, while Document A gets penalized down to `0.5`.

---

## 4. The 3 Scoring Signals (Step-by-Step Procedure)

To calculate the final cross-score between `0.0` and `1.0`:

### Signal 1: Coverage (50% weight)
- Count how many unique query words exist anywhere in the document.
- $\text{coverage} = \frac{\text{matching unique words}}{\text{total unique query words}}$
- **Rule:** If coverage is `0.0` (none of the query words exist in the document), stop and return `0.0`.

### Signal 2: Contiguous Bigrams (30% weight)
- Check how many pairs of consecutive words from the query appear consecutively in the document.
- For query `["deep", "learning", "python"]`, the pairs are:
  1. `("deep", "learning")`
  2. `("learning", "python")`
- Count how many of those pairs appear side-by-side in the document.
- If the query only has 1 word, set this to `1.0` if the word is in the document, else `0.0`.

### Signal 3: Proximity Density (20% weight)
- Find where each matching query word first appears in the document.
- $\text{span} = \text{last match index} - \text{first match index} + 1$.
- $\text{proximity} = \frac{\text{number of matching words}}{\text{span}}$.
- If all words appear right next to each other, $\text{span} == \text{count}$, so $\text{proximity} = 1.0$.

### Final Combined Score:
$$\text{final\_score} = \text{round}(0.5 \times \text{coverage} + 0.3 \times \text{ngram} + 0.2 \times \text{proximity}, 4)$$
