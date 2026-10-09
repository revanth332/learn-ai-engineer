# Hands-On AI Engineering: From First Principles to Production

A rigorous, production-grade learning repository designed to master core LLM, Agentic, Retrieval, and ML engineering concepts by implementing them from absolute scratch. 

Each module bypasses third-party wrappers (like LangChain, LlamaIndex, or Tiktoken) to implement underlying algorithms from scratch, complete with automated verification and a Coursera-style progress dashboard.

---

## 🛠️ Repository Architecture & Feature Map

```
.
├── src/
│   ├── tokenization/            # Tokenizers, context management, and costs
│   │   ├── bpe.py               # Byte-level BPE from scratch (train, encode, decode)
│   │   ├── context.py           # Newest-first context budget and truncation
│   │   ├── cost.py              # Pricing calculation engines
│   │   └── streaming.py         # Streaming token estimation
│   │
│   ├── retrieval/               # High-precision search & RAG engines
│   │   ├── vector_math.py       # Raw coordinate geometry (norm, cosine similarity)
│   │   ├── vector_store.py      # In-memory nearest-neighbor dense index
│   │   ├── bm25.py              # Okapi BM25 sparse keyword retrieval
│   │   ├── hybrid.py            # Reciprocal Rank Fusion (RRF) combiner
│   │   ├── reranking.py         # Cross-Encoder re-ranker stage
│   │   └── query_transform.py   # Multi-query decomposition, HyDE, and synonyms
│   │
│   ├── agent/                   # Agent runtimes & conversation state
│   │   ├── tools.py             # Dynamic tool schema & sandbox runner
│   │   └── memory.py            # TokenBuffer, Summarized, and Entity Memory
│   │
│   ├── structured_output.py     # Pydantic schema-guided parser & syntax repair
│   └── resilience.py            # Token Bucket Rate Limiting & Backoff with Jitter
│
├── tests/                       # Absolute unit-test verification suite (145+ passed)
└── dashboard.html               # Interactive visual engineering dashboard
```

---

## 🚀 Key Modules Implemented

### 1. Core Tokenization & Context Budgeting
*   **Byte-Level BPE**: A native Python implementation of Byte Pair Encoding. Supports vocabulary training from raw text, encoding characters as UTF-8 bytes to handle arbitrary characters, and deterministic decoding.
*   **Context Truncation**: A sliding newest-first context window that keeps chat prompts within target LLM token budgets.
*   **Pricing Estimators**: Strict execution cost calculation based on variable prompt/completion model tariffs.

### 2. Structured Outputs & System Resilience
*   **JSON Syntax Repair**: Fallback regex and AST parser that fixes half-truncated or malformed JSON payloads returned by LLMs.
*   **Resiliency Patterns**: 
    *   **Token Bucket**: Rate limiter enforcing strict request-per-minute (RPM) and token-per-minute (TPM) boundaries.
    *   **Exponential Backoff with Jitter**: Robust retry pattern for handling rate limits (`429` status codes) without overlapping retries.

### 3. Precision Information Retrieval (RAG Tier 1)
*   **Raw Vector Math**: Native distance and similarity calculations (cosine, dot-product, Euclidean, L2-normalization) without relying on numpy.
*   **Okapi BM25 Sparse Index**: Inverse Document Frequency (IDF) mapping using the Robertson-Spärck Jones formula to match key-phrases.
*   **Hybrid RRF Engine**: Merges dense semantic vector search with sparse keyword search using Reciprocal Rank Fusion (RRF).
*   **Cross-Encoder Re-ranker**: A second-stage re-ranking module utilizing semantic-lexical scoring factors.
*   **Query Transformation**: Decomposes complex prompts, generates Hypothetical Document Embeddings (HyDE), and expands synonyms.

### 4. Agentic Architecture & Conversation State
*   **Dynamic Tool Calling Engine**: Automatic JSON-schema generation from native Python type hints. Safely parses, validates, and executes tool arguments inside a sandboxed environment.
*   **Stateful Memory Systems**:
    *   **TokenBufferMemory**: Keeps conversations within token limitations via FIFO eviction.
    *   **SummaryMemory**: Compresses long conversation histories using a rolling summarizing strategy.
    *   **EntityMemory**: Dynamically parses context strings (e.g. `User is Developer`) and extracts entity properties to inject when queried.
    *   **CompositeMemory**: Orchestrates memory layers (system prompts, extracted entity context, sliding buffers) into unified OpenAI-compatible payloads.

---

## 📈 Dashboard & Verification

All implemented features are paired with explicit, granular assertions. Running the tests updates an interactive HTML dashboard:

```bash
# Run pytest to execute the full first-principles suite
python -m uv run python -m pytest -q

# Generate/Update the visual dashboard
python generate_dashboard.py
```

Open `dashboard.html` in your browser to view your verified learning progress.

---

## ⚙️ Requirements & Tech Stack
*   **Python**: 3.10+
*   **Runtime Manager**: `uv` (recommended for lighting fast package execution)
*   **Key Packages**: `pydantic`, `pytest`
