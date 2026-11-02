import asyncio
import json
import pytest
from src.agent.react import AgentState, StepTrace, AgentResult, AsyncReACTAgent


def _make_tc(name: str, args: dict | str, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "function": {
            "name": name,
            "arguments": json.dumps(args) if isinstance(args, dict) else args,
        },
    }


def test_react_fsm_transitions_and_completion():
    turn = 0

    async def mock_llm(messages, schemas):
        nonlocal turn
        turn += 1
        if turn == 1:
            return {
                "content": "I need to add 7 and 8.",
                "tool_calls": [_make_tc("add", {"a": 7, "b": 8}, "c1")],
            }
        # Verify observation from step 1 is present
        assert any(m.get("role") == "tool" and m.get("content") == "15" for m in messages)
        return {"content": "The result is 15.", "tool_calls": []}

    def add(a: int, b: int) -> int:
        return a + b

    agent = AsyncReACTAgent(llm_fn=mock_llm, tools={"add": add}, max_steps=4)
    result = asyncio.run(agent.run("What is 7 + 8?"))

    assert isinstance(result, AgentResult)
    assert result.final_state == AgentState.COMPLETED
    assert result.answer == "The result is 15."
    assert result.terminated_reason == "completed"
    assert result.state_history == [
        AgentState.IDLE,
        AgentState.THINKING,
        AgentState.ACTING,
        AgentState.OBSERVING,
        AgentState.THINKING,
        AgentState.COMPLETED,
    ]
    assert len(result.steps) == 2
    assert result.steps[0].observations == ["15"]


def test_react_bounded_parallel_execution_with_semaphore():
    active_count = 0
    peak_concurrency = 0

    async def slow_fetch(region: str) -> str:
        nonlocal active_count, peak_concurrency
        active_count += 1
        peak_concurrency = max(peak_concurrency, active_count)
        await asyncio.sleep(0.04)
        active_count -= 1
        return f"data-{region}"

    turn = 0

    def mock_llm(messages, schemas):
        nonlocal turn
        turn += 1
        if turn == 1:
            return {
                "content": "Fetching 4 regions in parallel.",
                "tool_calls": [
                    _make_tc("slow_fetch", {"region": "us-east"}, "c1"),
                    _make_tc("slow_fetch", {"region": "us-west"}, "c2"),
                    _make_tc("slow_fetch", {"region": "eu-central"}, "c3"),
                    _make_tc("slow_fetch", {"region": "ap-south"}, "c4"),
                ],
            }
        return {"content": "Fetched all 4 regions.", "tool_calls": []}

    agent = AsyncReACTAgent(
        llm_fn=mock_llm,
        tools={"slow_fetch": slow_fetch},
        max_concurrency=2,
    )
    result = asyncio.run(agent.run("Fetch all regions"))

    assert result.final_state == AgentState.COMPLETED
    assert peak_concurrency == 2
    assert result.steps[0].observations == [
        "data-us-east",
        "data-us-west",
        "data-eu-central",
        "data-ap-south",
    ]


def test_react_cycle_detection_with_reordered_dict_keys():
    turn = 0

    def mock_llm(messages, schemas):
        nonlocal turn
        turn += 1
        if turn == 1:
            # First call with keys (a, b)
            return {
                "content": "Step 1",
                "tool_calls": [_make_tc("multiply", '{"a": 3, "b": 4}', "c1")],
            }
        # Second call with same args but reversed key order (b, a)
        return {
            "content": "Step 2 repeating exact same call",
            "tool_calls": [_make_tc("multiply", '{"b": 4, "a": 3}', "c2")],
        }

    agent = AsyncReACTAgent(
        llm_fn=mock_llm,
        tools={"multiply": lambda a, b: a * b},
        max_repeat_actions=1,
    )
    result = asyncio.run(agent.run("Multiply 3 and 4"))

    assert result.final_state == AgentState.FAILED
    assert result.terminated_reason == "cycle_detected"
    assert result.state_history[-1] == AgentState.FAILED


def test_react_tool_timeout_and_missing_tool_resilience():
    async def hanging_tool() -> str:
        await asyncio.sleep(1.0)
        return "never"

    turn = 0

    def mock_llm(messages, schemas):
        nonlocal turn
        turn += 1
        if turn == 1:
            return {
                "content": "Calling hanging and unknown tools",
                "tool_calls": [
                    _make_tc("hanging_tool", {}, "c1"),
                    _make_tc("missing_tool", {}, "c2"),
                ],
            }
        return {"content": "Recovered from tool failures.", "tool_calls": []}

    agent = AsyncReACTAgent(
        llm_fn=mock_llm,
        tools={"hanging_tool": hanging_tool},
        tool_timeout=0.05,
    )
    result = asyncio.run(agent.run("Run flaky tools"))

    assert result.final_state == AgentState.COMPLETED
    obs = result.steps[0].observations
    assert "timed out" in obs[0]
    assert "not found" in obs[1]


def test_react_max_steps_fallback_synthesis():
    def mock_llm(messages, schemas):
        if schemas:
            # Keep calling a tool with unique arguments each step
            step_idx = len(messages)
            return {
                "content": "Still working...",
                "tool_calls": [_make_tc("echo", {"val": step_idx}, f"c_{step_idx}")],
            }
        return {"content": "Synthesized fallback after max steps.", "tool_calls": []}

    agent = AsyncReACTAgent(
        llm_fn=mock_llm,
        tools={"echo": lambda val: f"echo:{val}"},
        tool_schemas=[{"type": "function", "function": {"name": "echo"}}],
        max_steps=2,
    )
    result = asyncio.run(agent.run("Do a long task"))

    assert result.final_state == AgentState.FAILED
    assert result.terminated_reason == "Max steps reached!"
    assert result.answer == "Synthesized fallback after max steps."
    assert len(result.steps) == 2


def test_react_llm_quota_error_handling():
    def failing_llm(messages, schemas):
        raise RuntimeError("429 Resource Exhausted: quota exceeded")

    agent = AsyncReACTAgent(llm_fn=failing_llm, tools={})
    result = asyncio.run(agent.run("Hello"))

    assert result.final_state == AgentState.FAILED
    assert "Quota exceeded" in result.terminated_reason
