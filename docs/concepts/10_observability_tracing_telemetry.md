# Module 3.3: Observability, Tracing & Telemetry Engine

## 1. Why Traditional APM Fails for LLM & Agent Systems
In traditional web backends, observability tracks HTTP status codes (`200 OK` vs `500 Error`) and endpoint latency.
In multi-stage RAG and multi-agent systems, an endpoint can return `200 OK` while silently failing in production:
- **Hidden Cost Spikes**: A single ReAct loop or multi-agent handoff makes 8 LLM calls, burning $0.45 on a single user query without anyone knowing which sub-span caused the token explosion.
- **Latency Bottlenecks**: Total latency is 6.2 seconds, but is it due to dense embedding generation, cross-encoder reranking, slow tool execution, or LLM generation?
- **Streaming UX Blindness**: Total response time might be 4 seconds, while **Time-To-First-Token (TTFT)** is 280ms. Without tracking TTFT separately from total generation latency, you cannot optimize perceived responsiveness.
- **Async Context Loss**: When parallel workers (`asyncio.gather`) execute simultaneously, simple global variables lose track of which parent span spawned which child tool/LLM call.

Production AI observability platforms (OpenTelemetry GenAI Semantic Conventions, LangSmith, Arize Phoenix, Helicone) solve this using **Context-Propagated Hierarchical Trace Trees** and **Percentile Telemetry Aggregators**.

---

## 2. Core Architectural Concepts

### A. Traces vs. Spans (Hierarchical Tree)
- **Trace**: Represents the entire end-to-end execution of one user request (identified by `trace_id`).
- **Span**: Represents a single timed unit of work inside a trace (identified by `span_id` and linked to its parent via `parent_id`).
  - **Root Span** (`parent_id = None`): e.g., `"agent.run"` (0ms -> 1850ms)
    - **Child Span 1** (`parent_id = root.span_id`): `"retrieval.hybrid"` (5ms -> 120ms)
    - **Child Span 2** (`parent_id = root.span_id`): `"llm.generate"` (125ms -> 1400ms, `ttft_ms = 210.0`, `prompt_tokens = 1200`, `completion_tokens = 180`, `cost_usd = 0.0042`)
      - **Grandchild Span** (`parent_id = child2.span_id`): `"tool.sql_query"` (900ms -> 1150ms)

### B. Async-Safe Parent Propagation via `contextvars.ContextVar`
Why not store `self.current_span_id` on the `Tracer` instance?
Because if two async tasks (`asyncio.gather(worker_a(), worker_b())`) share the same `Tracer`, they will overwrite `self.current_span_id` and attach `worker_a`'s tool calls to `worker_b`'s span!

Python's built-in `contextvars.ContextVar` solves this automatically:
- Each async task gets its own isolated copy of the `ContextVar`.
- When entering a `with tracer.start_span("name") as span:` block:
  1. Read the current active span ID from `_active_span_var.get()`. That becomes `parent_id`.
  2. Set the new span's ID as active: `token = _active_span_var.set(span.span_id)`.
  3. When exiting the `with` block (`finally:`), restore the previous parent: `_active_span_var.reset(token)`.

### C. Time-To-First-Token (TTFT) & Token Throughput
For streaming LLM calls, two metrics govern user experience:
1. **TTFT (`ttft_ms`)**: Time elapsed from the start of the LLM request until the first non-empty token chunk arrives:
   `ttft_ms = (first_token_timestamp - start_time) * 1000.0`
2. **Token Throughput (`tokens_per_sec`)**:
   `tokens_per_sec = completion_tokens / (duration_ms / 1000.0)` (guarded against `duration_ms <= 0`).

### D. Latency Percentiles (P50, P95, P99)
Averages (`mean`) lie in production because a few outlier tool timeouts get hidden by fast cache hits.
To compute the `p`-th percentile (where `p` is in `[0, 100]`, e.g., `50`, `95`, `99`) over `N` sorted latencies `sorted_vals`:
1. If `sorted_vals` is empty, return `0.0`.
2. Compute the zero-based rank index using the nearest-rank / linear index formula:
   `idx = min(len(sorted_vals) - 1, max(0, math.ceil((p / 100.0) * len(sorted_vals)) - 1))`
   return `float(sorted_vals[idx])`

---

## 3. Step-by-Step Implementation Blueprint (`src/evals/observability.py`)

### Step 1: Imports & ContextVar
```python
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import math
import time
import uuid
from typing import Any, Iterator

_active_span_id_var: ContextVar[str | None] = ContextVar("_active_span_id", default=None)
```

### Step 2: `Span` Dataclass
```python
@dataclass
class Span:
    span_id: str
    trace_id: str
    name: str
    parent_id: str | None = None
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    status: str = "ok"  # "ok" or "error"
    error_message: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    ttft_ms: float | None = None
```
Add helper methods and properties on `Span`:
1. **`mark_first_token() -> None`**:
   - Records Time-To-First-Token in milliseconds relative to `self.start_time`:
   - `if self.ttft_ms is None:`
   - `self.ttft_ms = (time.perf_counter() - self.start_time) * 1000.0`
2. **`record_usage(prompt_tokens: int, completion_tokens: int, cost_usd: float = 0.0) -> None`**:
   - Accumulates `self.prompt_tokens += prompt_tokens`, `self.completion_tokens += completion_tokens`, `self.cost_usd += cost_usd`.
3. **`set_attribute(key: str, value: Any) -> None`**:
   - Sets `self.attributes[key] = value`.
4. **`@property def duration_ms(self) -> float`**:
   - If `self.end_time is None`, return `0.0`.
   - Return `max(0.0, (self.end_time - self.start_time) * 1000.0)`.
5. **`@property def total_tokens(self) -> int`**:
   - Return `self.prompt_tokens + self.completion_tokens`.
6. **`@property def tokens_per_sec(self) -> float`**:
   - Let `dur_sec = self.duration_ms / 1000.0`.
   - If `dur_sec <= 0.0` or `self.completion_tokens <= 0`, return `0.0`.
   - Return `self.completion_tokens / dur_sec`.

### Step 3: `TraceSummary` Dataclass
Represents the aggregated roll-up of an entire trace tree:
```python
@dataclass
class TraceSummary:
    trace_id: str
    total_spans: int
    total_duration_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cost_usd: float
    error_count: int
    span_breakdown: dict[str, float]  # span.name -> cumulative duration_ms
```

### Step 4: `Tracer` Class
Manages all spans belonging to a `trace_id`:
1. **`__init__(self, trace_id: str | None = None)`**:
   - `self.trace_id = trace_id or uuid.uuid4().hex`
   - `self._spans: list[Span] = []`
2. **`@contextmanager def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[Span]`**:
   - Read `parent_id = _active_span_id_var.get()`.
   - Create `span = Span(span_id=uuid.uuid4().hex[:16], trace_id=self.trace_id, name=name, parent_id=parent_id, attributes=dict(attributes or {}))`.
   - Append `span` to `self._spans`.
   - Set active span context: `token = _active_span_id_var.set(span.span_id)`.
   - Wrap `yield span` in `try ... except Exception as exc: ... finally:`:
     - In `except Exception as exc:`:
       - Set `span.status = "error"`
       - Set `span.error_message = str(exc)`
       - Re-raise `exc` (`raise`) so errors are never swallowed!
     - In `finally:`:
       - Set `span.end_time = time.perf_counter()`
       - Reset context var: `_active_span_id_var.reset(token)`
3. **`get_spans(self) -> list[Span]`**:
   - Return a defensive copy `list(self._spans)`.
4. **`get_children(self, parent_span_id: str) -> list[Span]`**:
   - Return `[s for s in self._spans if s.parent_id == parent_span_id]`.
5. **`summarize(self) -> TraceSummary`**:
   - Compute:
     - `total_spans = len(self._spans)`
     - Root spans are `[s for s in self._spans if s.parent_id is None]`.
     - `total_duration_ms = sum(s.duration_ms for s in root_spans)` (if no root spans exist, `0.0`).
     - `total_prompt_tokens = sum(s.prompt_tokens for s in self._spans)`
     - `total_completion_tokens = sum(s.completion_tokens for s in self._spans)`
     - `total_tokens = total_prompt_tokens + total_completion_tokens`
     - `total_cost_usd = round(sum(s.cost_usd for s in self._spans), 8)`
     - `error_count = sum(1 for s in self._spans if s.status == "error")`
     - `span_breakdown`: dictionary mapping each `span.name` to the sum of `span.duration_ms` for spans with that name.
   - Return `TraceSummary(...)`.

### Step 5: `TelemetryAggregator` Class (Multi-Trace Fleet Metrics & Percentiles)
Aggregates metrics across many traces in production to compute P50/P95/P99 latency SLAs, TTFT percentiles, error rates, and total fleet cost:
1. **`__init__(self)`**:
   - `self._spans: list[Span] = []`
2. **`record_trace(self, tracer: Tracer) -> None`**:
   - `self._spans.extend(tracer.get_spans())`
3. **`@staticmethod def compute_percentile(values: list[float], percentile: float) -> float`**:
   - If `not values`: return `0.0`.
   - Sort values ascending: `sorted_vals = sorted(values)`.
   - Compute index: `idx = min(len(sorted_vals) - 1, max(0, math.ceil((percentile / 100.0) * len(sorted_vals)) - 1))`
   - Return `float(sorted_vals[idx])`.
4. **`get_latency_percentiles(self, span_name: str | None = None) -> dict[str, float]`**:
   - Filter spans: if `span_name` is provided, take `[s.duration_ms for s in self._spans if s.name == span_name]`, else take `[s.duration_ms for s in self._spans]`.
   - Return:
     ```python
     {
         "p50": self.compute_percentile(durations, 50.0),
         "p95": self.compute_percentile(durations, 95.0),
         "p99": self.compute_percentile(durations, 99.0),
     }
     ```
5. **`get_ttft_percentiles(self) -> dict[str, float]`**:
   - Filter spans that recorded TTFT: `ttfts = [s.ttft_ms for s in self._spans if s.ttft_ms is not None]`.
   - Return `{"p50": ..., "p95": ..., "p99": ...}` using `self.compute_percentile`.
6. **`get_fleet_metrics(self) -> dict[str, Any]`**:
   - If `not self._spans`: return zeros for `total_spans`, `error_rate`, `total_tokens`, `total_cost_usd`, and `latency_ms` (`{"p50": 0.0, "p95": 0.0, "p99": 0.0}`).
   - Compute:
     - `total_spans = len(self._spans)`
     - `error_count = sum(1 for s in self._spans if s.status == "error")`
     - `error_rate = error_count / total_spans`
     - `total_tokens = sum(s.total_tokens for s in self._spans)`
     - `total_cost_usd = round(sum(s.cost_usd for s in self._spans), 8)`
     - `latency_ms = self.get_latency_percentiles()`
   - Return dictionary with keys `"total_spans"`, `"error_rate"`, `"total_tokens"`, `"total_cost_usd"`, `"latency_ms"`.
