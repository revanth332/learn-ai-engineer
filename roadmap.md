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

4. **LLM Evaluation, IR Metrics & Guardrails (Evals)**
   - Deterministic Information Retrieval (IR) metrics: Recall@K, Precision@K, Mean Reciprocal Rank (MRR), and NDCG.
   - Production RAG Triad: NLI Atomic Claim Faithfulness, Reverse-Query Embedding Answer Relevance, and Context Relevance.
   - Deterministic Guardrails: PII regex detection/redaction and prompt-injection/jailbreak firewalls.
   - Systematic evaluation harness with concurrent execution (`asyncio`) and benchmark reporting.

5. **Tier 2 Capstone: End-to-End Stateful RAG & Tool Pipeline**
   - Full integration wiring Query Transformation -> Hybrid Search (BM25 + Vector RRF) -> Cross-Encoder Reranking -> Composite Memory -> Dynamic Tool Registry -> Input/Output Guardrails with a live LLM.

---

## Tier 3: Advanced Architectures & Production Systems (Hard)
Focus: Async autonomy, observability, low-latency scaling, model internals, and fine-tuning.

1. **Async Agentic Loops from First Principles (ReAct State Machine)**
   - Explicit ReAct (Reasoning + Acting) state machine built on `asyncio`.
   - Parallel tool execution (`asyncio.gather` with `Semaphore` rate control).
   - Dynamic replanning, cyclic execution, and deterministic deadlock/loop breakers.

2. **Multi-Agent Coordination & Shared Blackboard**
   - Supervisor-worker routing models and handoff protocols.
   - Shared blackboard state store with optimistic concurrency and conflict resolution.

3. **Observability, Tracing & Telemetry Engine**
   - Hierarchical trace & span collectors (OpenTelemetry pattern for LLMs).
   - Latency profiling: Time-To-First-Token (TTFT), Inter-Token Latency (ITL), P95/P99 latency distributions, and per-span cost attribution.

4. **Semantic Caching & High-Concurrency Optimization**
   - Vector similarity cache with calibrated cosine distance thresholds.
   - Cache invalidation strategies (TTL, LRU) and hybrid exact-match/semantic routing.
   - Async request batching and queue management.

5. **Model Internals, Sampling Math & Local Inference**
   - Logit manipulation from scratch: Softmax, Temperature scaling, Top-K, and Top-P (Nucleus) sampling.
   - KV caching memory mechanics, context length scaling, and quantization trade-offs (GGUF, AWQ).
   - Local serving backends (vLLM, Ollama) and raw prompt formatting (ChatML, Llama-3).

6. **Fine-Tuning & Synthetic Data Pipelines**
   - Synthetic dataset generation with LLM rejection sampling and quality filtering.
   - Instruction-tuning and preference (DPO) format preparation.
   - LoRA/QLoRA rank/alpha architecture and pre/post-tuning benchmark evaluation.

---

## Tier 4: Full-Stack AI Engineering & Deployment (Production Capstone)
Focus: Bridging Python AI backends with real-time Node/FastAPI streaming and React interfaces.

1. **Production Streaming AI Backend (FastAPI / Node + SSE)**
   - Multiplexed Server-Sent Events (SSE) streaming tokens, tool execution states, and citations.
   - Client disconnect propagation (`AbortController`), request cancellation, and idempotency keys.

2. **Full-Stack AI Agent Interface (React)**
   - Real-time token streaming renderer with debounced search and modular card architecture.
   - Live tool-call status inspection, retrieved source citation cards, and trace latency telemetry.
