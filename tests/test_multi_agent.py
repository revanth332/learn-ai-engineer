import asyncio
import pytest

from src.agent.multi_agent import (
    BlackboardEntry,
    MultiAgentResult,
    SharedBlackboard,
    SupervisorOrchestrator,
    WorkerAgent,
    WorkerOutput,
)


def test_shared_blackboard_concurrency_and_defensive_copies():
    async def _run():
        board = SharedBlackboard()

        async def writer_task(agent_id: int):
            await asyncio.sleep(0.005)
            await board.write(f"key_{agent_id}", agent_id * 10, author=f"worker_{agent_id}")

        await asyncio.gather(*(writer_task(i) for i in range(10)))

        snapshot = board.get_snapshot()
        assert len(snapshot) == 10
        assert snapshot["key_5"] == 50
        assert board.read("key_9") == 90
        assert board.read("non_existent", default="fallback") == "fallback"

        # Mutating snapshot or history must not mutate internal blackboard state
        snapshot["key_5"] = 9999
        snapshot["hacked"] = True
        assert board.read("key_5") == 50
        assert board.read("hacked") is None

        history = board.get_history()
        assert len(history) == 10
        assert all(isinstance(entry, BlackboardEntry) for entry in history)
        history.clear()
        assert len(board.get_history()) == 10

    asyncio.run(_run())


def test_worker_agent_sync_and_async_handlers():
    async def _run():
        board = SharedBlackboard()
        await board.write("topic", "Vector Databases", author="system")

        def sync_researcher(task: str, state: dict) -> WorkerOutput:
            return WorkerOutput(
                agent_name="researcher",
                content=f"Researched {state['topic']} for task: {task}",
                state_updates={"findings": ["HNSW", "IVF"]},
                handoff_to="writer",
            )

        async def async_writer(task: str, state: dict) -> WorkerOutput:
            await asyncio.sleep(0.01)
            findings = ", ".join(state.get("findings", []))
            return WorkerOutput(
                agent_name="writer",
                content=f"Drafted article covering {findings}",
                state_updates={"draft_ready": True},
                handoff_to=None,
            )

        w1 = WorkerAgent("researcher", "Finds facts", sync_researcher)
        w2 = WorkerAgent("writer", "Writes drafts", async_writer)

        out1 = await w1.execute("Write technical post", board)
        assert out1.handoff_to == "writer"
        assert board.read("findings") == ["HNSW", "IVF"]

        out2 = await w2.execute("Write technical post", board)
        assert out2.content == "Drafted article covering HNSW, IVF"
        assert board.read("draft_ready") is True

    asyncio.run(_run())


def test_supervisor_run_parallel_fan_out():
    async def _run():
        board = SharedBlackboard()

        async def security_worker(task: str, state: dict) -> WorkerOutput:
            await asyncio.sleep(0.02)
            return WorkerOutput(
                agent_name="security",
                content="No SQLi found",
                state_updates={"security_passed": True},
            )

        async def perf_worker(task: str, state: dict) -> WorkerOutput:
            await asyncio.sleep(0.02)
            return WorkerOutput(
                agent_name="perf",
                content="Latency under 10ms",
                state_updates={"perf_passed": True},
            )

        orchestrator = SupervisorOrchestrator(
            workers=[
                WorkerAgent("security", "Security audit", security_worker),
                WorkerAgent("perf", "Performance audit", perf_worker),
            ]
        )

        outputs = await orchestrator.run_parallel(
            "Audit PR #42", ["security", "perf"], blackboard=board
        )
        assert len(outputs) == 2
        assert board.read("security_passed") is True
        assert board.read("perf_passed") is True
        assert len(board.get_history()) == 2

    asyncio.run(_run())


def test_supervisor_sequential_handoffs_and_synthesizer():
    async def _run():
        def planner_fn(task: str, state: dict) -> WorkerOutput:
            return WorkerOutput(
                agent_name="planner",
                content="Plan created",
                state_updates={"plan": ["step1", "step2"]},
                handoff_to="coder",
            )

        async def coder_fn(task: str, state: dict) -> WorkerOutput:
            await asyncio.sleep(0.01)
            return WorkerOutput(
                agent_name="coder",
                content=f"Implemented {len(state['plan'])} steps",
                state_updates={"code_written": True},
                handoff_to="reviewer",
            )

        def reviewer_fn(task: str, state: dict) -> WorkerOutput:
            return WorkerOutput(
                agent_name="reviewer",
                content="LGTM!",
                state_updates={"approved": True},
                handoff_to=None,
            )

        async def custom_synthesizer(
            task: str, final_state: dict, trace: list[WorkerOutput]
        ) -> str:
            await asyncio.sleep(0.005)
            return f"Task '{task}' approved={final_state.get('approved')} after {len(trace)} stages."

        orchestrator = SupervisorOrchestrator(
            workers=[
                WorkerAgent("planner", "Plans work", planner_fn),
                WorkerAgent("coder", "Writes code", coder_fn),
                WorkerAgent("reviewer", "Reviews code", reviewer_fn),
            ],
            synthesizer_fn=custom_synthesizer,
            max_steps=5,
        )

        result = await orchestrator.run("Build Auth API", initial_state={"env": "prod"})
        assert isinstance(result, MultiAgentResult)
        assert result.terminated_reason == "completed"
        assert result.handoff_chain == ["planner", "coder", "reviewer"]
        assert result.blackboard_state["env"] == "prod"
        assert result.blackboard_state["approved"] is True
        assert result.final_answer == "Task 'Build Auth API' approved=True after 3 stages."

    asyncio.run(_run())


def test_supervisor_handoff_cycle_detection():
    async def _run():
        def agent_a_fn(task: str, state: dict) -> WorkerOutput:
            return WorkerOutput(
                agent_name="agent_a",
                content="Passing to B",
                state_updates={"a_visited": True},
                handoff_to="agent_b",
            )

        def agent_b_fn(task: str, state: dict) -> WorkerOutput:
            return WorkerOutput(
                agent_name="agent_b",
                content="Passing back to A",
                state_updates={"b_visited": True},
                handoff_to="agent_a",
            )

        orchestrator = SupervisorOrchestrator(
            workers=[
                WorkerAgent("agent_a", "Agent A", agent_a_fn),
                WorkerAgent("agent_b", "Agent B", agent_b_fn),
            ],
            max_steps=10,
        )

        result = await orchestrator.run("Ping pong task")
        assert result.terminated_reason == "handoff_cycle_detected"
        assert result.handoff_chain == ["agent_a", "agent_b", "agent_a"]

    asyncio.run(_run())


def test_supervisor_invalid_target_and_max_steps():
    async def _run():
        def bad_handoff_fn(task: str, state: dict) -> WorkerOutput:
            return WorkerOutput(
                agent_name="worker_1",
                content="Handing off to ghost",
                handoff_to="non_existent_worker",
            )

        orchestrator_invalid = SupervisorOrchestrator(
            workers=[WorkerAgent("worker_1", "Worker 1", bad_handoff_fn)],
            max_steps=5,
        )
        res_invalid = await orchestrator_invalid.run("Test invalid target")
        assert res_invalid.terminated_reason == "invalid_handoff_target"

        # Test max_steps_exceeded with linear chain longer than max_steps
        w1 = WorkerAgent(
            "w1", "W1", lambda t, s: WorkerOutput("w1", "step 1", handoff_to="w2")
        )
        w2 = WorkerAgent(
            "w2", "W2", lambda t, s: WorkerOutput("w2", "step 2", handoff_to="w3")
        )
        w3 = WorkerAgent(
            "w3", "W3", lambda t, s: WorkerOutput("w3", "step 3", handoff_to=None)
        )

        orchestrator_max = SupervisorOrchestrator(workers=[w1, w2, w3], max_steps=2)
        res_max = await orchestrator_max.run("Long chain")
        assert res_max.terminated_reason == "max_steps_exceeded"
        assert res_max.handoff_chain == ["w1", "w2"]

    asyncio.run(_run())
