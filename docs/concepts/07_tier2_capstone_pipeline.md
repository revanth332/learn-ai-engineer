# Module 2.5: Tier 2 Capstone — End-to-End Stateful RAG & Tool Pipeline

## 1. Why This Capstone Exists

Up to this point, you have built every core subsystem of a modern AI Engineering stack in isolation:
1. **Query Transformation (`query_transform.py`)**: Synonym expansion, multi-query decomposition, and query fusion.
2. **Hybrid Retrieval (`hybrid.py`)**: Dense Vector Search + Sparse BM25 keyword search fused via Reciprocal Rank Fusion (RRF).
3. **Stage-2 Cross-Encoder Reranking (`reranking.py`)**: High-precision scoring of top candidate chunks.
4. **Stateful Conversation & Memory (`memory.py`)**: Token-budgeted sliding windows, rolling summaries, and entity memory (`CompositeMemory`).
5. **Dynamic Tool Execution (`tools.py`)**: Automatic JSON schema introspection and runtime tool execution (`ToolRegistry`).
6. **Input/Output Guardrails (`guardrails.py`)**: Prompt injection/jailbreak firewall (`is_safe`) and PII redaction (`redact`, `detect`).

In production at companies like Stripe, Notion, or Perplexity, an AI Engineer's job is to **wire these isolated primitives into a single, fault-tolerant, stateful orchestration loop**.

---

## 2. Production Pipeline Architecture (7-Stage Lifecycle)

Every user request flows through a strict, deterministic 7-stage lifecycle:

```text
[User Query]
     |
     v
Stage 1: Input Guardrail (Pre-Flight Firewall)
     |-- is_safe(query) -> If False, short-circuit immediately & block request!
     |-- redact(query)  -> Scrub any accidental PII before sending to external APIs
     |
     v
Stage 2: Query Transformation
     |-- expand_query_synonyms(clean_query, synonym_map)
     |-- decompose_query(clean_query)
     |-- fuse_transformed_queries(clean_query, expanded, sub_queries)
     |
     v
Stage 3: Multi-Query Hybrid Retrieval (Dense + BM25 + RRF)
     |-- For each transformed query:
     |      query_vec = embedding_fn(q)
     |      hits = hybrid_retriever.search(q, query_vec, top_k=top_k)
     |-- Deduplicate candidate documents by doc_id (storing raw text from metadata)
     |
     v
Stage 4: Stage-2 Cross-Encoder Reranking
     |-- Convert unique hits into CandidateDocument(id, text, metadata)
     |-- rerank(clean_query, candidates, top_n=rerank_top_n) -> top RankedDocuments
     |
     v
Stage 5: Stateful Context Assembly + Tool Loop
     |-- Retrieve conversation history: memory.get_chat_history(clean_query)
     |-- Inject retrieved context chunks into the augmented prompt
     |-- Call LLM (with registered ToolRegistry schemas if tools are available)
     |-- If LLM requests a tool call -> execute via tool_registry.execute(...) -> feed tool result back to LLM
     |
     v
Stage 6: Output Guardrail (Post-Flight Firewall)
     |-- is_safe(raw_answer) -> If unsafe, replace with fallback refusal message
     |-- redact(raw_answer)  -> Scrub any leaked PII before returning to user
     |
     v
Stage 7: Memory Persistence & Telemetry
     |-- memory.add_message("user", clean_query)
     |-- memory.add_message("assistant", safe_answer)
     |-- Return structured PipelineResponse
```

---

## 3. Core Data Structures & Interfaces

### 3.1 `PipelineResponse` (Dataclass)
Every turn of the pipeline should return a transparent, inspectable result object (essential for debugging, observability, and running `EvaluationHarness` benchmarks):

```python
@dataclass
class PipelineResponse:
    query: str                        # Original or PII-redacted user query
    answer: str                       # Final safe, PII-redacted assistant answer
    retrieved_context: list[str]      # Final reranked document texts injected into prompt
    tool_calls: list[dict[str, Any]]  # Record of any tools executed during this turn
    blocked: bool = False             # True if blocked by pre-flight or post-flight guardrails
    block_reason: list[str] = field(default_factory=list)
```

### 3.2 Standardized LLM Callable Contract
To keep the pipeline testable offline (with deterministic mocks) and runnable online (with OpenAI / OpenRouter), define a clean LLM callable signature:

```python
# llm_fn takes (messages: list[dict[str, Any]], tools: list[dict[str, Any]])
# and returns a dict representing the assistant response:
# {
#     "content": "Final text answer (or None if calling a tool)",
#     "tool_calls": [
#         {"id": "call_1", "name": "calculator", "arguments": {"a": 5, "b": 3}}
#     ]  # optional or empty list if no tool calls
# }
```

---

## 4. Step-by-Step Implementation Blueprint

You will create `src/agent/pipeline.py` (or `src/pipeline.py`) with two main parts:

### Part 1: Document Ingestion Helper (`ingest_document`)
Wait — `HybridRetriever.add(doc_id, text, vector, metadata)` stores `metadata` inside `VectorStore`, and `HybridResult` returns `(doc_id, score, metadata)`.
Notice that `CandidateDocument` in `reranking.py` needs the document's `text`!
- **Critical Architectural Insight:** When you ingest a document into `HybridRetriever`, store `{"text": text, **(metadata or {})}` inside `metadata`! That way, when `hybrid_retriever.search(...)` returns `HybridResult(doc_id, score, metadata)`, `result.metadata["text"]` is immediately available to pass into `CandidateDocument(id=result.doc_id, text=result.metadata["text"], metadata=result.metadata)` for Stage-2 Reranking!

### Part 2: `StatefulRAGAgent` Class

#### Constructor (`__init__`)
Accepts and wires the components:
- `hybrid_retriever: HybridRetriever`
- `memory: CompositeMemory`
- `tool_registry: ToolRegistry`
- `llm_fn: Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]`
- `embedding_fn: Callable[[str], list[float]]`
- `synonym_map: dict[str, list[str]] | None = None`
- `retrieval_top_k: int = 5`
- `rerank_top_n: int = 3`
- `max_tool_iterations: int = 3`

#### Method 1: `ingest(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> None`
1. Compute `vector = self.embedding_fn(text)`.
2. Build `full_meta = {**(metadata or {}), "text": text}`.
3. Call `self.hybrid_retriever.add(doc_id, text, vector, full_meta)`.

#### Method 2: `retrieve_and_rerank(self, query: str) -> list[RankedDocument]`
1. **Transform query**:
   - `expanded = expand_query_synonyms(query, self.synonym_map or {})`
   - `sub_queries = decompose_query(query)`
   - `queries = fuse_transformed_queries(query, expanded, sub_queries)`
2. **Multi-query Hybrid Search & Deduplication**:
   - Maintain a dict `candidates_by_id: dict[str, CandidateDocument] = {}`.
   - For each `q` in `queries`:
     - `q_vec = self.embedding_fn(q)`
     - `results = self.hybrid_retriever.search(q, q_vec, top_k=self.retrieval_top_k)`
     - For each `res` in `results`:
       - If `res.doc_id` not in `candidates_by_id`:
         - Extract `doc_text = res.metadata.get("text", "")`
         - Store `CandidateDocument(id=res.doc_id, text=doc_text, metadata=res.metadata)`
3. **Stage-2 Cross-Encoder Reranking**:
   - Return `rerank(query, list(candidates_by_id.values()), top_n=self.rerank_top_n)`

#### Method 3: `run(self, user_query: str) -> PipelineResponse`
1. **Stage 1 — Pre-Flight Guardrails**:
   - Check `safe, reasons = is_safe(user_query)`.
   - If not `safe`:
     - Return `PipelineResponse(query=user_query, answer="Request blocked by security guardrails.", retrieved_context=[], tool_calls=[], blocked=True, block_reason=reasons)`
   - Scrub PII from input: `clean_query = redact(user_query)`.
2. **Stage 2, 3, 4 — Retrieve & Rerank**:
   - `ranked_docs = self.retrieve_and_rerank(clean_query)`
   - `context_texts = [doc.text for doc in ranked_docs]`
   - `formatted_context = "\n\n".join(context_texts)`
3. **Stage 5 — Context Assembly & Tool Execution Loop**:
   - Get prior chat history: `messages = list(self.memory.get_chat_history(clean_query))`
   - Append the current augmented user message:
     ```python
     augmented_content = (
         f"Context:\n{formatted_context}\n\nQuestion: {clean_query}"
         if formatted_context
         else clean_query
     )
     messages.append({"role": "user", "content": augmented_content})
     ```
   - Prepare tool schemas: `tool_schemas = self.tool_registry.get_schemas()`
   - Loop up to `self.max_tool_iterations`:
     - `response = self.llm_fn(messages, tool_schemas)`
     - `requested_tools = response.get("tool_calls") or []`
     - If not `requested_tools`:
       - `raw_answer = response.get("content", "")`
       - Break out of the loop!
     - Otherwise, for each `tc` in `requested_tools`:
       - Record `tc` in `executed_tool_calls`.
       - Execute tool: `exec_result = self.tool_registry.execute(tc["name"], tc["arguments"])`
       - Append tool message to `messages`: `messages.append(exec_result.to_chat_message(tool_call_id=tc.get("id")))`
   - If loop finishes without setting `raw_answer`, call `self.llm_fn(messages, [])` one final time to get `raw_answer = response.get("content", "")`.
4. **Stage 6 — Post-Flight Output Guardrails**:
   - Check `ans_safe, ans_reasons = is_safe(raw_answer)`.
   - If not `ans_safe`:
     - Return `PipelineResponse(query=clean_query, answer="Response blocked by output safety guardrails.", retrieved_context=context_texts, tool_calls=executed_tool_calls, blocked=True, block_reason=ans_reasons)`
   - Scrub any leaked PII: `safe_answer = redact(raw_answer)`
5. **Stage 7 — Memory Persistence**:
   - `self.memory.add_message("user", clean_query)`
   - `self.memory.add_message("assistant", safe_answer)`
   - Return `PipelineResponse(query=clean_query, answer=safe_answer, retrieved_context=context_texts, tool_calls=executed_tool_calls, blocked=False, block_reason=[])`
