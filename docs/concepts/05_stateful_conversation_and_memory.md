# Stateful Conversation & Memory Architecture

## 1. Why Conversation Memory Matters
LLMs are inherently stateless: every request to `/chat/completions` is independent and retains zero recollection of previous turns. To simulate continuity, client systems maintain conversational state and reinject history into each prompt.

However, naive history injection creates two critical failure modes:
1. **Context Window Saturation**: Rapidly exceeding max context bounds or spending prohibitive token costs.
2. **Context Degradation & Distraction**: Older, irrelevant conversational details dilute the LLM's attention, causing the "lost in the middle" phenomenon and degraded reasoning.

---

## 2. Core Memory Architectures

### A. Sliding Token-Window Memory (`TokenBufferMemory`)
- Maintains raw message history in strict FIFO order up to a maximum token threshold.
- When budget is exceeded, oldest turns are evicted until the payload fits within budget.
- System prompt instructions are permanently pinned and exempt from eviction.

### B. Summary Memory & Recursive Rolling Summarizer (`SummaryMemory`)
- Maintains a compact running natural-language summary of past conversational context.
- When message count or tokens exceed a threshold, older turns are compressed into the running summary:
  `New Summary = Summarize(Old Summary + Old Messages)`
- Prompt structure:
  `[System Prompt] -> [Current Summary] -> [Recent Messages Buffer]`

### C. Entity Memory & Knowledge Graph (`EntityMemory`)
- Extracts specific entities (people, technologies, dates, user preferences) and relational facts `(subject, predicate, object)` from conversation turns.
- Stores entities in a lightweight graph or lookup table.
- Dynamically injects only relevant entity facts into the prompt when referenced by recent turns.

---

## 3. Real-World Architecture Pattern

```
                       User Turn
                          │
                          ▼
            ┌───────────────────────────┐
            │  Entity Fact Extraction   │ ──► Update Entity Graph
            └─────────────┬─────────────┘
                          ▼
             Is Token Budget Exceeded?
             ├──► Yes: Trigger Rolling Summarizer
             │         Compress oldest turns into Running Summary
             └──► No: Keep in Working Buffer
                          │
                          ▼
           Assemble LLM Request Payload:
     [Pinned System Prompt]
   + [Extracted Entity Context]
   + [Condensed Rolling Summary]
   + [Active Window Messages]
```

---

## 4. Implementation Plan (`src/agent/memory.py`)
1. `Message`: Structured chat message model (`role`, `content`, `token_count`, `timestamp`).
2. `TokenBufferMemory`: Token-budgeted FIFO buffer with pinned system messages.
3. `SummaryMemory`: Running summary compaction with token accounting.
4. `EntityMemory`: Lightweight key-value / triple entity extraction store.
5. `CompositeMemory`: Unified multi-strategy memory controller orchestrating entity extraction, rolling summaries, and sliding message buffers.
