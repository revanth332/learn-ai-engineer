# AI Engineering Curriculum & Roadmap

This roadmap focuses on building production-grade AI systems from first principles, avoiding superficial wrapper abstractions.

---

## Tier 1: Foundations (Easy)
Focus: Token mechanics, raw API patterns, deterministic control, and baseline retrieval.

1. **Tokenization & Context Math**
   - Byte-Pair Encoding (BPE) mechanics.
   - Context budget calculation and dynamic trimming.
   - Token-to-cost modeling.
   - Raw streaming response handling and generator pipelines.

2. **Structured Outputs & Schema Enforcement**
   - Enforcing JSON output via strict schemas and function calling specifications.
   - Deterministic parser validation, repair algorithms, and Pydantic integration without high-level wrapper libraries.

3. **Resilience & Rate Limiting**
   - Exponential backoff with jitter.
   - Fallback routing across models/providers.
   - Circuit breakers for downstream availability.
   - Token-bucket and sliding-window rate limiters.

4. **Vector Math & In-Memory Retrieval**
   - Raw vector math: dot product, cosine similarity, Euclidean distance, normalization.
   - In-memory vector index with exact nearest neighbor search.
   - Document chunking algorithms: fixed-size, sentence-boundary, recursive character.

---

## Tier 2: Intermediate Systems (Medium)
Focus: Multi-stage pipelines, stateful interactions, and production-grade retrieval.

1. **Advanced RAG Pipelines**
   - Hybrid Search: Sparse (BM25) + Dense (embeddings) retrieval fused via Reciprocal Rank Fusion (RRF).
   - Cross-Encoder Re-ranking: Re-scoring top-k candidates to eliminate retrieval noise.
   - Query Transformation: HyDE (Hypothetical Document Embeddings), query expansion, and multi-query decomposition.

2. **Dynamic Tool Calling & Runtime Engine**
   - Tool registry design and dynamic schema generation.
   - Argument validation and execution sandboxing.
   - Error-recovery feedback loops when the model outputs invalid parameters or fails tool execution.

3. **Stateful Conversation & Memory Architecture**
   - Buffer memory and token-budgeted sliding window memory.
   - Entity memory extraction and graph representation.
   - Recursive rolling summarization.

4. **LLM Evaluation & Guardrails (Evals)**
   - Implementing the RAG Triad: Context Relevance, Groundedness, and Answer Relevance.
   - Deterministic checks (regex, schema, semantic similarity) vs. LLM-as-a-judge patterns.
   - Systematic evaluation harness and benchmark reporting.

---

## Tier 3: Advanced Architectures (Hard)
Focus: Autonomy, dynamic control flow, low-latency scaling, and fine-tuning.

1. **Agentic Loops from First Principles**
   - ReAct (Reasoning + Acting) pattern via explicit state machines.
   - Dynamic replanning, cyclic execution, and deadlock detection.
   - Multi-agent coordination: Supervisor-worker models and shared blackboard state.

2. **Semantic Caching & Latency Optimization**
   - Vector similarity cache with calibrated distance thresholds.
   - Cache invalidation strategies and hybrid exact-match/semantic routing.
   - Batching and asynchronous queue management.

3. **Local Inference & Serving Mechanics**
   - Serving via local backends (vLLM, Ollama).
   - Prompt formatting specifications (ChatML, Llama-3 format).
   - KV caching dynamics, context length scaling, and quantization trade-offs (GGUF, AWQ).

4. **Fine-Tuning Pipelines**
   - Data curation and synthetic dataset generation with rejection sampling.
   - Instruction-tuning format preparation.
   - LoRA/QLoRA configuration and evaluation benchmarks against base models.
