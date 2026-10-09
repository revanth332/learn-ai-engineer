# Machine Learning Foundations for AI Engineering

This document outlines the core ML concepts directly relevant to AI engineering and tracks the implementation progress across our curriculum.

---

## 1. Core ML Concepts Required for AI Engineering

### A. Sampling Mechanics, Logits, and Output Distributions [In Progress]
- **Logits**: Raw, unnormalized score vectors output by the transformer over the vocabulary for the next token position.
- **Softmax Function**: Converts logits into a normalized categorical probability distribution:
  sigma(z)_i = exp(z_i / T) / sum(exp(z_j / T))
- **Temperature (T)**:
  - T -> 0 (Greedy / Argmax): Sharply favors the top logit, making generation deterministic.
  - High T: Flattens the distribution, increasing the entropy and diversity of token selection.
- **Top-k Sampling**: Truncates the sampling pool to the k most probable tokens before sampling.
- **Top-p (Nucleus) Sampling**: Selects the minimal subset of tokens whose cumulative probability >= p, dynamically adapting the pool width based on prediction confidence.
- **Hallucination Dynamics**: Structural byproduct of probabilistic autoregressive sampling; foundation models maximize statistical plausibility, not factual truth.
- **Current Status**: BPE tokenization, token budget trimming, and streaming protocols implemented. Sampling / logit temperature scaling module will be added.

### B. High-Dimensional Vector Geometry & Embeddings [Completed]
- **Embeddings**: Dense numerical vectors projecting semantic meaning into high-dimensional space (e.g., 768, 1536, 3072 dimensions).
- **Curse of Dimensionality**: In high dimensions, spatial volume expands exponentially; pairwise distances between random vectors concentrate near orthogonality.
- **Distance & Similarity Metrics**:
  - **Dot Product**: dot(a, b) = sum(a_i * b_i)
  - **Cosine Similarity**: dot(a, b) / (norm(a) * norm(b)) (measures directional alignment independent of magnitude).
  - **Euclidean Distance (L2)**: sqrt(sum((a_i - b_i)^2))
  - Note: When embeddings are L2-normalized (norm(a) = 1), Cosine Similarity and Dot Product are mathematically identical.
- **Vector Search Mechanics**: Exact flat search (O(num_documents * embedding_dimension)) vs. Approximate Nearest Neighbors (ANN) via graph structures (HNSW) or inverted files (IVF).
- **Current Status**: Fully implemented in `src/retrieval/` (Vector math, in-memory cosine Vector Store, Okapi BM25, Reciprocal Rank Fusion, Stage-2 Cross-Encoder Reranking, and Multi-Query / HyDE transformations).

### C. Transformer Operational Mechanics [Upcoming]
- **Self-Attention**: Attention computation via Query (Q), Key (K), and Value (V) projections:
  Attention(Q, K, V) = softmax((Q * K_transpose) / sqrt(d_k)) * V
- **KV Caching**: Storing Key and Value states of past tokens across transformer layers to prevent redundant O(sequence_length^2) computation during autoregressive generation. Directly bounds memory usage for high-concurrency LLM serving.
- **Positional Dynamics**: Methods like RoPE (Rotary Position Embeddings) and attention decay leading to context-window edge degradation ("lost in the middle").
- **Current Status**: Scheduled for the Local Inference & Serving module.

### D. Retrieval and Generative Metrics [Upcoming]
- **Information Retrieval Metrics**: Precision@K, Recall@K, Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (NDCG).
- **Evaluation**: Perplexity (exp(cross_entropy_loss)), RAG Triad (Context Relevance, Groundedness, Answer Relevance), and LLM-as-a-judge alignment.
- **Current Status**: Scheduled for Tier 3 (Evaluation & Observability Engine).

---

## 2. Non-Essential Concepts (Avoid Spending Time Here)
- Deriving manual backpropagation gradients or manual SGD/Adam optimizers from scratch.
- Classical tabular ML algorithms (SVMs, Decision Trees, K-Means, XGBoost) unless building hybrid classical pipelines.
- Training foundation models from scratch on distributed clusters.
- Low-level GPU CUDA kernel optimization (unless writing custom inference runtimes).

---

## 3. Execution Strategy
ML concepts are implemented directly alongside system code rather than studied in isolation:
- Implement vector math from first principles when constructing retrieval indexes.
- Measure token distributions, logits, and temperature scaling directly in tokenization and API layers.
- Implement retrieval metrics (Precision@K, MRR) when building eval harnesses.
