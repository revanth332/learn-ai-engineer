# Tier 3, Module 3.2: Multi-Agent Coordination & Shared Blackboard

## 1. Why Single Agents Fail at Scale

When you give a single ReAct agent 40 tools and a 2,000-line system prompt covering SQL queries, Python code execution, security auditing, and customer billing:
1. **Tool Selection Degradation**: LLM tool-calling accuracy drops sharply once a single agent has more than 10–15 tools with overlapping schemas.
2. **Context Window Pollution**: Intermediate scratchpad traces from a 10-step SQL exploration flood the context window, causing the LLM to forget the original security constraint.
3. **Instruction Conflict**: A "creative copywriter" persona and a "strict compliance auditor" persona directly contradict each other inside a single system prompt.

Production systems (like LangGraph, OpenAI Swarm, and Microsoft AutoGen) solve this by splitting responsibilities across **Specialized Worker Agents** coordinated via two core mechanisms:
- **Shared Blackboard Architecture**: A concurrency-safe central state store where agents read context and publish structured findings with full provenance (who wrote what, and when).
- **Supervisor Routing & Explicit Agent Handoffs**: Deterministic control transfer between specialists with ping-pong loop protection.

---

## 2. Core Terminology & Production Patterns

### 2.1 The Shared Blackboard Pattern (`SharedBlackboard`)
Originated in classic distributed AI (HEARSAY-II) and modernized in LangGraph `StateGraph`:
- Instead of passing giant, unstructured chat histories back and forth between every pair of agents (`O(num_agents ^ 2)` message complexity), all agents read from and write to a single **Shared Blackboard**.
- **Concurrency Safety (`asyncio.Lock`)**: Because multiple async workers may write to the blackboard simultaneously during `asyncio.gather`, mutations to the blackboard are protected by an `asyncio.Lock`.
- **Provenance Tracking (`BlackboardEntry`)**: Every state update records the `author` (agent name), `key`, `value`, and `step` so downstream agents (and observability tools) know which specialist produced which artifact.

### 2.2 Supervisor-Worker Topology
```text
                   +----------------------------+
                   |   SupervisorOrchestrator   |
                   | (Routes, Parallelizes,     |
                   |  & Synthesizes Final Out)  |
                   +-------------+--------------+
                                 |
          +----------------------+----------------------+
          |                      |                      |
          v                      v                      v
+-------------------+  +-------------------+  +-------------------+
|  ResearcherAgent  |  |    CoderAgent     |  |   ReviewerAgent   |
+---------+---------+  +---------+---------+  +---------+---------+
          |                      |                      |
          +----------------------+----------------------+
                                 |
                                 v
                   +----------------------------+
                   |      SharedBlackboard      |
                   | (asyncio.Lock protected    |
                   |  kv state + history log)   |
                   +----------------------------+
```

### 2.3 Explicit Agent-to-Agent Handoffs (Swarm Pattern)
Sometimes a worker discovers mid-task that another specialist is needed (e.g., `TriageAgent` realizes a ticket is a billing dispute and hands off directly to `BillingAgent`).
- A worker can return a `WorkerOutput` containing `handoff_to="billing_agent"`.
- **Ping-Pong Loop Breaker (`max_handoffs`)**: If `Agent A` hands off to `Agent B` and `Agent B` hands right back to `Agent A`, an unguarded swarm enters an infinite handoff cycle. The orchestrator enforces `max_handoffs` (and tracks handoff chains) to guarantee termination.

---

## 3. Real-World Example

Suppose an enterprise user submits:
> *"Research our Q3 churn metrics, write a Python script to forecast Q4 churn, and audit the script for security vulnerabilities."*

1. **Supervisor Routing**:
   - The `SupervisorOrchestrator` inspects the user query and registered workers (`researcher`, `coder`, `security_auditor`).
   - It dispatches `researcher` first (or runs independent workers in parallel via `run_parallel`).
2. **Blackboard Publication**:
   - `researcher` writes `blackboard.write(key="q3_metrics", value="Churn was 4.2%", author="researcher")`.
3. **Handoff / Sequential Chaining**:
   - `researcher` triggers a handoff to `coder` (`handoff_to="coder"`).
   - `coder` reads `blackboard.get_snapshot()` (seeing `q3_metrics: Churn was 4.2%`), generates the forecast script, writes `key="forecast_script"` to the blackboard, and hands off to `security_auditor`.
4. **Consensus / Final Synthesis**:
   - `security_auditor` verifies the script, writes `key="security_verdict"`, and finishes (`handoff_to=None`).
   - `SupervisorOrchestrator` synthesizes the final response from the blackboard artifacts and returns a `MultiAgentResult`.

---

## 4. Step-by-Step Implementation Procedure

You will create `src/agent/multi_agent.py` with 4 clean components:

### Step 1: Data Structures
1. **`BlackboardEntry` (Dataclass)**:
   - `key: str`
   - `value: Any`
   - `author: str` (which agent wrote this entry)
   - `timestamp: float = field(default_factory=time.time)`

2. **`WorkerOutput` (Dataclass)**:
   - `agent_name: str`
   - `content: str` (the textual output/findings from this worker)
   - `state_updates: dict[str, Any] = field(default_factory=dict)` (key-value pairs to write to the blackboard)
   - `handoff_to: str | None = None` (optional name of the next worker agent to hand off control to)

3. **`MultiAgentResult` (Dataclass)**:
   - `final_answer: str`
   - `blackboard_state: dict[str, Any]`
   - `execution_trace: list[WorkerOutput]` (ordered list of all worker outputs executed)
   - `handoff_chain: list[str]` (ordered list of agent names that ran, e.g. `["researcher", "coder"]`)
   - `terminated_reason: str` (`"completed"`, `"max_steps_exceeded"`, `"invalid_handoff_target"`, `"handoff_cycle_detected"`)

---

### Step 2: `SharedBlackboard` Class
A concurrency-safe state store shared across all agents in a run:
- **`__init__(self)`**:
  - `self._state: dict[str, Any] = {}`
  - `self._history: list[BlackboardEntry] = []`
  - `self._lock = asyncio.Lock()`
- **`async write(self, key: str, value: Any, author: str) -> None`**:
  - Acquire `async with self._lock:`:
    - Set `self._state[key] = value`
    - Append `BlackboardEntry(key=key, value=value, author=author)` to `self._history`.
- **`async update_many(self, updates: dict[str, Any], author: str) -> None`**:
  - Acquire `async with self._lock:`:
    - For `k, v` in `updates.items()`: update `self._state[k] = v` and append `BlackboardEntry(k, v, author)` to `self._history`.
- **`read(self, key: str, default: Any = None) -> Any`**:
  - Return `self._state.get(key, default)`.
- **`get_snapshot(self) -> dict[str, Any]`**:
  - Return `dict(self._state)` (a shallow copy so callers cannot accidentally mutate internal state without the lock).
- **`get_history(self) -> list[BlackboardEntry]`**:
  - Return `list(self._history)`.

---

### Step 3: `WorkerAgent` Class
Represents a specialized domain agent:
- **`__init__(self, name: str, role_description: str, handler_fn: Callable[[str, dict[str, Any]], Awaitable[WorkerOutput | dict[str, Any]] | WorkerOutput | dict[str, Any]])`**:
  - `self.name = name`
  - `self.role_description = role_description`
  - `self.handler_fn = handler_fn`
- **`async execute(self, task: str, blackboard: SharedBlackboard) -> WorkerOutput`**:
  1. Read current state snapshot: `snapshot = blackboard.get_snapshot()`.
  2. Call `raw = self.handler_fn(task, snapshot)`. If `inspect.isawaitable(raw)`, `raw = await raw`.
  3. Normalize `raw` into a `WorkerOutput`:
     - If `isinstance(raw, WorkerOutput)`: `output = raw`
     - Else (if `raw` is a `dict`):
       ```python
       output = WorkerOutput(
           agent_name=self.name,
           content=raw.get("content") or "",
           state_updates=raw.get("state_updates") or {},
           handoff_to=raw.get("handoff_to"),
       )
       ```
  4. If `output.state_updates`:
     - Call `await blackboard.update_many(output.state_updates, author=self.name)`.
  5. Also write the worker's latest output under `f"{self.name}_output"` or let `state_updates` handle explicit keys. (Writing `state_updates` via `update_many` keeps the blackboard clean and explicit).
  6. Return `output`.

---

### Step 4: `SupervisorOrchestrator` Class
Coordinates the workers, handles routing/handoffs, parallel execution, and final synthesis:
- **`__init__(self, workers: list[WorkerAgent], router_fn: Callable[[str, dict[str, Any], list[str]], Awaitable[str | None] | str | None] | None = None, synthesizer_fn: Callable[[str, dict[str, Any], list[WorkerOutput]], Awaitable[str] | str] | None = None, max_steps: int = 5)`**:
  - Store workers in a dict `self.workers: dict[str, WorkerAgent] = {w.name: w for w in workers}`.
  - Store `self.router_fn`, `self.synthesizer_fn`, and `self.max_steps`.

- **`async run_parallel(self, task: str, worker_names: list[str], blackboard: SharedBlackboard | None = None) -> list[WorkerOutput]`**:
  - Executes multiple independent workers concurrently on the same `SharedBlackboard` using `asyncio.gather`:
    ```python
    board = blackboard or SharedBlackboard()
    tasks = [self.workers[name].execute(task, board) for name in worker_names if name in self.workers]
    return list(await asyncio.gather(*tasks))
    ```

- **`async run(self, query: str, initial_state: dict[str, Any] | None = None) -> MultiAgentResult`**:
  1. Create a fresh `blackboard = SharedBlackboard()` for this request (concurrency-safe per call!).
  2. If `initial_state`: call `await blackboard.update_many(initial_state, author="system")`.
  3. Initialize `execution_trace: list[WorkerOutput] = []`, `handoff_chain: list[str] = []`, `visited_handoffs: set[tuple[str, str]] = set()`.
  4. Determine starting worker `current_worker_name: str | None = None`:
     - If `self.router_fn` is provided: call it with `(query, blackboard.get_snapshot(), list(self.workers.keys()))` (awaiting if needed).
     - Else: default to the first registered worker `next(iter(self.workers.keys()), None)`.
  5. Loop `for step in range(self.max_steps):`
     - If `not current_worker_name` or `current_worker_name.upper() == "FINISH"`:
       - Break out of loop with `terminated_reason = "completed"`.
     - If `current_worker_name not in self.workers`:
       - Return/break with `terminated_reason = "invalid_handoff_target"`.
     - Execute worker:
       ```python
       worker = self.workers[current_worker_name]
       handoff_chain.append(current_worker_name)
       output = await worker.execute(query, blackboard)
       execution_trace.append(output)
       ```
     - **Determine next worker (Handoff vs Router)**:
       - If `output.handoff_to`:
         - Check for ping-pong cycle: `edge = (current_worker_name, output.handoff_to)`
         - If `edge in visited_handoffs` or `output.handoff_to == current_worker_name`:
           - Break with `terminated_reason = "handoff_cycle_detected"`.
         - `visited_handoffs.add(edge)`
         - `current_worker_name = output.handoff_to`
       - Else if `self.router_fn`:
         - Ask `self.router_fn(query, blackboard.get_snapshot(), list(self.workers.keys()))` who should run next (or `"FINISH"` / `None`).
       - Else:
         - No handoff and no router -> work is complete! Break with `terminated_reason = "completed"`.
  6. If the `for` loop finished all `self.max_steps` without breaking:
     - Set `terminated_reason = "max_steps_exceeded"`.
  7. **Synthesize Final Answer**:
     - If `self.synthesizer_fn` is provided: call it with `(query, blackboard.get_snapshot(), execution_trace)` (awaiting if needed).
     - Else: join or return the last worker's content (`execution_trace[-1].content if execution_trace else ""`).
  8. Return `MultiAgentResult(...)`.
