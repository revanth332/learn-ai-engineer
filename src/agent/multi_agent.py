from dataclasses import dataclass, field
from typing import Any,Callable,Awaitable
import time
import asyncio
import inspect

@dataclass
class BlackboardEntry:
    key:str
    value:str
    author:str
    timestamp:float = field(default_factory=time.time)

@dataclass
class WorkerOutput:
    agent_name:str
    content:str
    state_updates:dict[str,Any] = field(default_factory=dict)
    handoff_to:str | None = None

@dataclass
class MultiAgentResult:
    final_answer:str
    blackboard_state:dict[str,Any]
    execution_trace:list[WorkerOutput]
    handoff_chain:list[str]
    terminated_reason:str

class SharedBlackboard:
    def __init__(self):
        self._state: dict[str,Any] = {}
        self._history: list[BlackboardEntry] = []
        self._lock = asyncio.Lock()

    async def write(self,key:str,value:Any,author:str) -> None:
        async with self._lock:
            self._state[key] = value
            self._history.append(BlackboardEntry(key,value,author))

    async def update_many(self,updates:dict[str,Any],author:str) -> None:
        async with self._lock:
            for key,value in updates.items():
                self._state[key] = value
                self._history.append(BlackboardEntry(key,value,author))

    def read(self,key:str,default:Any = None) -> Any:
        return self._state.get(key,default)

    def get_snapshot(self) -> dict[str,Any]:
        return dict(self._state)

    def get_history(self) -> list[BlackboardEntry]:
        return list(self._history)

class WorkerAgent:
    def __init__(self,name:str,description:str,handler_fn:Callable[[str,dict[str,Any]],Awaitable[WorkerOutput | dict[str,Any]] | WorkerOutput | dict[str,Any]]):
        self.name = name
        self.description = description
        self.handler_fn = handler_fn

    async def execute(self,task:str,blackboard:SharedBlackboard) -> WorkerOutput:
        snapshot = blackboard.get_snapshot()
        response = self.handler_fn(task,snapshot)
        if inspect.isawaitable(response):
            response = await response
        if isinstance(response,WorkerOutput):
            output = response
        else:
            output = WorkerOutput(self.name,response.get("content") or "",response.get("state_updates") or {},response.get("handoff_to"))

        if output.state_updates:
            await blackboard.update_many(output.state_updates,self.name)
        return output

class SupervisorOrchestrator:
    def __init__(self,workers: list[WorkerAgent],router_fn:Callable[[str,dict[str,Any],list[str]],Awaitable[str | None] | str | None]=None,synthesizer_fn:Callable[[str,dict[str,Any],list[WorkerOutput]],Awaitable[str] | str] | None=None,max_steps:int=5):
        self.router_fn = router_fn
        self.synthesizer_fn = synthesizer_fn
        self.workers = {w.name : w for w in workers}
        self.max_steps = max_steps

    async def run_parallel(self,task:str,worker_names:list[str],blackboard:dict[str,Any] | None = None) -> list[WorkerOutput]:
        board = blackboard or SharedBlackboard()
        tasks = [self.workers[name].execute(task,board) for name in worker_names if name in self.workers]
        return list(await asyncio.gather(*tasks))

    async def run(self,task:str,initial_state:dict[str,Any] | None=None) -> MultiAgentResult:
        board = SharedBlackboard()
        if initial_state:
            await board.update_many(initial_state,"system")
        execution_trace: list[WorkerOutput] = []
        handoff_chain: list[str] = []
        visited_handoffs:set[tuple[str,str]] = set()
        current_worker_name:str | None = None

        if self.router_fn:
            worker_name = self.router_fn(task,board.get_snapshot(),list(self.workers.keys()))
            if inspect.isawaitable(worker_name):
                worker_name = await worker_name
        else:
            worker_name = next(iter(self.workers.keys()),None)

        current_worker_name = worker_name
        for step in range(self.max_steps+1):
            if step == self.max_steps:
                terminated_reason = "max_steps_exceeded"
                break
            if current_worker_name is None or current_worker_name.lower() == "finish":
                terminated_reason = "completed"
                break
            if current_worker_name not in self.workers:
                terminated_reason = "invalid_handoff_target"
                break
            handoff_chain.append(current_worker_name)
            worker_output = await self.workers[current_worker_name].execute(task,board)
            execution_trace.append(worker_output)
            if worker_output.handoff_to:
                edge = (current_worker_name,worker_output.handoff_to)
                if edge in visited_handoffs or worker_output.handoff_to == current_worker_name:
                    terminated_reason = "handoff_cycle_detected"
                    break
                visited_handoffs.add((current_worker_name,worker_output.handoff_to))
                current_worker_name = worker_output.handoff_to
            elif self.router_fn:
                current_worker_name = self.router_fn(task,board.get_snapshot(),list(self.workers.keys()))
                if inspect.isawaitable(current_worker_name):
                    current_worker_name = await current_worker_name
            else:
                terminated_reason = "completed"
                break

        if self.synthesizer_fn:
            final_answer = self.synthesizer_fn(task,board.get_snapshot(),execution_trace)
            if inspect.isawaitable(final_answer):
                final_answer = await final_answer
        else:
            final_answer = execution_trace[-1].content if execution_trace else ""
        return MultiAgentResult(final_answer,board.get_snapshot(),execution_trace,handoff_chain,terminated_reason)

