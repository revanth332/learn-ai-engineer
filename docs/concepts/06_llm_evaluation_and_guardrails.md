# Module 2.4: Production LLM Evaluation & Guardrails (Evals)

---

## 1. The Industry Standard Evaluation Stack

Production RAG systems (Ragas, TruLens, DeepEval) do not rely on toy token overlap. They use a three-tier evaluation architecture:

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 1: Deterministic Guardrails (Regex, PII, Blocklists)   │ -> Sub-millisecond pre/post flight filters
├─────────────────────────────────────────────────────────────┤
│ Tier 2: Embedding Cosine Projections (Reverse Query Gen)    │ -> Semantic vector alignment
├─────────────────────────────────────────────────────────────┤
│ Tier 3: LLM-as-a-Judge with NLI Claim Decomposition        │ -> Atomic proposition verification (0/1)
└─────────────────────────────────────────────────────────────┘
```

---

## 2. The RAG Triad: Production Algorithms

### A. Faithfulness / Groundedness (NLI Claim Extraction & Verification)

```
Generated Answer
      │
      ▼
[LLM Claim Extractor] ──> ["Claim 1", "Claim 2", "Claim 3"]
                                  │
                                  ▼
[NLI Judge: Context + Claim] ──> Verdicts: [1, 1, 0]
                                  │
                                  ▼
Score = sum(verdicts) / count(claims)
```

1. **Step 1 (Atomic Claim Extraction):**
   Prompt the LLM to decompose the answer into standalone propositions:
   ```json
   {
     "claims": [
       "PostgreSQL was released in 1996.",
       "PostgreSQL has native graph traversal indexing."
     ]
   }
   ```
2. **Step 2 (NLI Entailment Check):**
   For each claim, the Judge evaluates whether the context provides sufficient evidence:
   ```json
   {
     "claim": "PostgreSQL has native graph traversal indexing.",
     "verdict": 0,
     "reason": "Context does not mention native graph indexing."
   }
   ```
3. **Step 3 (Mathematical Score):**
   `faithfulness = count(supported_claims) / total_claims`

---

### B. Answer Relevance (Reverse-Query Embedding Projection)

Prevents models from deflecting or answering tangential questions:

```
Generated Answer
      │
      ▼
[LLM Query Generator] ──> [Gen_Q1, Gen_Q2, Gen_Q3]
                                  │
                                  ▼
[Embedding Model] ──> [Embed(Gen_Q1), Embed(Gen_Q2)]
                                  │
                                  ▼
CosineSimilarity(Embed(Gen_Q_i), Embed(Original_Query))
                                  │
                                  ▼
Score = average(cosine_similarities)
```

1. **Step 1:** Model generates 3 questions that the *answer alone* addresses.
2. **Step 2:** Compute vector embeddings for all generated questions and the original user query using `src/retrieval/vector_math.py`.
3. **Step 3:** Compute cosine similarity:
   `score = (1 / k) * sum(cosine_similarity(embed(gen_q_i), embed(orig_query)))`

---

### C. Context Relevance (Sentence-Level Information Density)

Measures signal-to-noise ratio in retrieved context:

1. Context is split into sentences `S = [s_1, s_2, ..., s_n]`.
2. The Judge LLM selects the subset of sentences `S_rel` strictly necessary to answer the query:
   ```json
   {
     "relevant_sentence_indices": [0, 2]
   }
   ```
3. **Mathematical Score:**
   `context_relevance = count(S_rel) / count(S)`

---

## 3. Production Guardrails Engine

Input/Output guardrails operate synchronously on incoming prompts and outgoing completions:

### PII Redaction:
- Scans for emails, phone numbers, credit cards, SSNs, and vendor API keys (`sk-...`, `ghp_...`, `AKIA...`).
- Replaces matches with deterministic tokens (`[REDACTED_EMAIL]`, `[REDACTED_API_KEY]`).

### Jailbreak & Policy Filters:
- Blocklist and regex engine scanning for prompt injection patterns (`ignore prior instructions`, `developer mode`, `system prompt override`).

---

## 4. Class Design & Data Contracts

```python
@dataclass
class EvalSample:
    query: str
    context: str
    answer: str
    ground_truth: str | None = None

@dataclass
class TriadScore:
    context_relevance: float
    groundedness: float
    answer_relevance: float
    details: dict[str, Any]

class RAGTriadEvaluator:
    def __init__(
        self,
        llm_fn: Callable[[str], str],
        embedding_fn: Callable[[str], list[float]],
    ) -> None:
        ...

    def evaluate_faithfulness(self, context: str, answer: str) -> tuple[float, list[dict]]:
        ...

    def evaluate_answer_relevance(self, query: str, answer: str) -> tuple[float, list[str]]:
        ...

    def evaluate_context_relevance(self, query: str, context: str) -> tuple[float, list[int]]:
        ...

    def evaluate_sample(self, sample: EvalSample) -> TriadScore:
        ...
```

