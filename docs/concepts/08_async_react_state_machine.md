# Tier 3, Module 3.1: Async Agentic Loops from First Principles (ReAct State Machine)

## 1. Why Synchronous `for` Loops Fail in Production Agent Systems

In Tier 2, your agent loop ran **synchronously**:
```text
Tool 1 (300ms) -> wait -> Tool 2 (400ms) -> wait -> Tool 3 (500ms) = 1,200ms total
```
In production AI systems (like Cursor, Devin, or Perplexity Pro Search), an LLM frequently emits **parallel tool calls** in a single reasoning step (e.g., querying 4 microservice APIs or searching 3 documentation indices simultaneously).
1. **Sequential Bottleneck**: Running 4 I/O-bound tools sequentially wastes 75% of wall-clock latency.
2. **Unbounded Concurrency Crash**: Conversely, firing 50 async tool calls with bare `asyncio.gather` without rate limiting triggers HTTP 429 Too Many Requests or exhausts database connection pools.
3. **Infinite Reasoning Loops**: Autonomous agents frequently get stuck calling the exact same tool with the exact same arguments (`search("python error")` -> `search("python error ")`) until they burn through the user's token budget.
4. **Opaque Control Flow**: A bare `while`/`for` loop hides the agent's lifecycle state, making it impossible to stream state transitions to a frontend UI or pause/resume execution.

To solve all four problems, production agent frameworks (LangGraph, AutoGen, OpenAI Swarm) model the agent as an **Explicit Asynchronous Finite State Machine (Async ReAct FSM)**.

---

## 2. Core Terminology & Production Patterns

### 2.1 The ReAct Paradigm (Reason + Act)
Introduced by Yao et al. (2022), **ReAct** interleaves explicit reasoning traces with external environment actions:
- **Thought (Reason)**: The LLM analyzes the goal, prior observations, and decides what information is missing.
- **Action (Act)**: The LLM emits one or more structured tool calls (`name`, `arguments`).
- **Observation (Observe)**: The runtime executes the tools concurrently and returns their outputs to the state history.
- **Synthesis / Termination**: Once enough observations are gathered (or a loop/step limit is hit), the machine transitions to `COMPLETED` or `FAILED`.

### 2.2 Explicit Finite State Machine (`AgentState`)
Instead of implicit loop variables, the agent explicitly transitions across well-defined states:
```text
[IDLE]
  |
  v
[THINKING]  <------------------------------------+
  |                                              |
  +---> (No tool calls / Final Answer) ---> [COMPLETED]
  |                                              |
  +---> (Tool calls emitted)                     |
          |                                      |
          v                                      |
      [ACTING] (Concurrent asyncio.gather        |
          |     bounded by asyncio.Semaphore)    |
          v                                      |
     [OBSERVING] (Append observations &          |
          |       check Loop/Cycle Breaker) -----+
          |
          +---> (Cycle detected or Max Steps) ---> [FAILED / FALLBACK]
```

### 2.3 Bounded Parallel Execution (`asyncio.Semaphore` + `asyncio.gather` + `asyncio.wait_for`)
When the LLM requests `N` tool calls in a single turn:
- Wrap each tool call in a semaphore context (`async with self.semaphore:`) so at most `max_concurrency` tools execute simultaneously.
- Enforce a per-tool execution timeout (`await asyncio.wait_for(..., timeout=self.tool_timeout)`) so a hanging third-party API never deadlocks the agent.
- Support **both** `async def` coroutines (`inspect.iscoroutinefunction`) and synchronous functions (either called directly or via `asyncio.to_thread`).

### 2.4 Deterministic Cycle / Loop Breaker
How do production agents detect that an LLM is stuck in a loop *before* wasting 10 turns?
- Compute a canonical **Action Signature** for each tool call:
  ```python
  signature = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
  ```
- Track how many times each `signature` has been invoked during the current run (`action_counts: dict[str, int]`).
- If `action_counts[signature] > max_repeat_actions` (e.g., the agent calls the exact same tool with the exact same arguments a 2nd or 3rd time), **trip the Loop Breaker**:
  - Inject a system/tool warning observation (`"LOOP DETECTED: You already called {tool_name} with these exact arguments. Do not repeat this action; synthesize your best answer now."`) or transition straight to synthesis.

---

## 3. Real-World Example

Suppose a user asks an Autonomous FinOps Agent:
> *"Compare our AWS EC2 spend in `us-east-1`, `us-west-2`, and `eu-central-1` for March."*

1. **State `IDLE` -> `THINKING` (Step 1)**:
   - LLM emits 3 parallel tool calls:
     - `get_aws_spend(region="us-east-1", month="March")`
     - `get_aws_spend(region="us-west-2", month="March")`
     - `get_aws_spend(region="eu-central-1", month="March")`
2. **State `THINKING` -> `ACTING`**:
   - The `AsyncReActAgent` runs all 3 calls concurrently via `asyncio.gather` gated by `asyncio.Semaphore(2)`.
   - Total latency is 2x single call latency instead of 3x.
3. **State `ACTING` -> `OBSERVING`**:
   - All 3 results are recorded into `StepTrace` and appended to the conversation history in deterministic order.
4. **State `OBSERVING` -> `THINKING` (Step 2)**:
   - LLM sees all 3 regional numbers and produces the final comparison table (`tool_calls = []`).
5. **State `THINKING` -> `COMPLETED`**:
   - Returns `AgentRunResult` containing the final answer, full state transition history, step traces, and total steps taken.

---

## 4. Step-by-Step Implementation Procedure

You will create `src/agent/react.py` with the following components:

### Step 1: Enum & Data Structures
1. **`AgentState(Enum)`**:
   - `IDLE = "idle"`
   - `THINKING = "thinking"`
   - `ACTING = "acting"`
   - `OBSERVING = "observing"`
   - `COMPLETED = "completed"`
   - `FAILED = "failed"`

2. **`StepTrace` (Dataclass)**:
   - `step_number: int`
   - `thought: str` (the `content` or reasoning emitted by the LLM during `THINKING`)
   - `tool_calls: list[dict[str, Any]]` (the actions requested in this step)
   - `observations: list[str]` (the string outputs returned by the tools in this step)

3. **`AgentRunResult` (Dataclass)**:
   - `answer: str`
   - `final_state: AgentState`
   - `steps: list[StepTrace]`
   - `state_history: list[AgentState]`
   - `terminated_reason: str` (e.g., `"completed"`, `"max_steps_exceeded"`, `"cycle_detected"`)

### Step 2: `AsyncReActAgent` Class

#### Constructor (`__init__`)
```python
class AsyncReActAgent:
    def __init__(
        self,
        llm_fn: Callable[[list[dict[str, Any]], list[dict[str, Any]]], Awaitable[dict[str, Any]] | dict[str, Any]],
        tools: dict[str, Callable[..., Any]],
        tool_schemas: list[dict[str, Any]] | None = None,
        max_steps: int = 5,
        max_concurrency: int = 3,
        max_repeat_actions: int = 1,
        tool_timeout: float = 5.0,
        system_prompt: str = "You are an autonomous ReAct agent.",
    ) -> None:
```
- Initialize `self.semaphore = asyncio.Semaphore(max_concurrency)`.
- Store all configuration attributes.

#### Helper 1: `_transition(self, new_state: AgentState, state_history: list[AgentState]) -> AgentState`
- Appends `new_state` to `state_history` and returns `new_state`.

#### Helper 2: `async _call_llm(self, messages: list[dict[str, Any]], schemas: list[dict[str, Any]]) -> dict[str, Any]`
- Calls `res = self.llm_fn(messages, schemas)`.
- If `inspect.isawaitable(res)`, `return await res`; otherwise `return res`. (This makes your agent work seamlessly with both `async def` and sync LLM functions!).

#### Helper 3: `async _execute_single_tool(self, name: str, arguments: dict[str, Any]) -> str`
- Acquire `async with self.semaphore:`:
  - Check if `name` is in `self.tools`. If not, return `f"Error: Tool '{name}' not found."`.
  - Call the tool inside `try ... except asyncio.TimeoutError` and `except Exception as e`:
    - If `inspect.iscoroutinefunction(fn)`:
      `result = await asyncio.wait_for(fn(**arguments), timeout=self.tool_timeout)`
    - Else:
      `result = await asyncio.wait_for(asyncio.to_thread(fn, **arguments), timeout=self.tool_timeout)`
    - Return `str(result)`
  - If `asyncio.TimeoutError`: return `f"Error: Tool '{name}' timed out after {self.tool_timeout}s."`
  - If any other `Exception as e`: return `f"Error executing '{name}': {e}"`

#### Main Loop: `async run(self, query: str) -> AgentRunResult`
1. Initialize `state_history: list[AgentState] = []`, `steps: list[StepTrace] = []`, `action_counts: dict[str, int] = {}`.
2. Transition to `AgentState.IDLE`.
3. Initialize `messages`:
   ```python
   messages = [
       {"role": "system", "content": self.system_prompt},
       {"role": "user", "content": query},
   ]
   ```
4. Loop `for step_num in range(1, self.max_steps + 1):`
   - **Transition to `AgentState.THINKING`**.
   - Call `response = await self._call_llm(messages, self.tool_schemas or [])`.
   - Extract `thought = response.get("content") or ""` and `requested_tools = response.get("tool_calls") or []`.
   - **Check Termination**: If `not requested_tools`:
     - Transition to `AgentState.COMPLETED`.
     - Append `StepTrace(step_num, thought, [], [])` to `steps`.
     - Return `AgentRunResult(answer=thought, final_state=AgentState.COMPLETED, steps=steps, state_history=state_history, terminated_reason="completed")`.
   - **Check Cycle / Loop Breaker before Acting**:
     - For each `tc` in `requested_tools`:
       - Compute `sig = f"{tc['name']}:{json.dumps(tc.get('arguments', {}), sort_keys=True)}"`
       - Increment `action_counts[sig] = action_counts.get(sig, 0) + 1`
       - If `action_counts[sig] > self.max_repeat_actions`:
         - Transition to `AgentState.FAILED`.
         - Return `AgentRunResult(answer=thought, final_state=AgentState.FAILED, steps=steps, state_history=state_history, terminated_reason="cycle_detected")`.
   - **Transition to `AgentState.ACTING`**:
     - Execute all `requested_tools` concurrently using `asyncio.gather`:
       ```python
       tasks = [
           self._execute_single_tool(tc["name"], tc.get("arguments", {}))
           for tc in requested_tools
       ]
       observations = list(await asyncio.gather(*tasks))
       ```
   - **Transition to `AgentState.OBSERVING`**:
     - Record `StepTrace(step_num, thought, requested_tools, observations)` in `steps`.
     - If `thought`: append `{"role": "assistant", "content": thought}` to `messages`.
     - For each `tc, obs` in `zip(requested_tools, observations)`:
       - Append `{"role": "tool", "name": tc["name"], "content": obs, "tool_call_id": tc.get("id", "")}` to `messages`.
5. **If the `for` loop exhausts `max_steps` without completing**:
   - Call `final_res = await self._call_llm(messages, [])` (no tools allowed) to get a best-effort synthesis `final_answer = final_res.get("content") or ""`.
   - Transition to `AgentState.FAILED`.
   - Return `AgentRunResult(answer=final_answer, final_state=AgentState.FAILED, steps=steps, state_history=state_history, terminated_reason="max_steps_exceeded")`.
