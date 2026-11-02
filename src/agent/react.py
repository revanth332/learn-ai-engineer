from enum import StrEnum
from dataclasses import dataclass
from typing import Any,Awaitable,Callable
import asyncio
import inspect
import json
from src.llm import call,get_git_status,get_git_diff
from src.agent.tools import ToolRegistry

class AgentState(StrEnum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class StepTrace:
    step_number:int
    thought:str
    tool_calls:list[dict[str,Any]]
    observations:list[str]

@dataclass
class AgentResult:
    answer:str
    final_state:AgentState
    state_history:list[AgentState]
    steps:list[StepTrace]
    terminated_reason:str

class AsyncReACTAgent:
    def __init__(self,
        llm_fn: Callable[[list[dict[str, Any]], list[dict[str, Any]]], Awaitable[dict[str, Any]] | dict[str, Any]],
        tools: dict[str, Callable[..., Any]],
        tool_schemas: list[dict[str, Any]] | None = None,
        max_steps: int = 5,
        max_concurrency: int = 3,
        max_repeat_actions: int = 1,
        tool_timeout: float = 5.0,
        system_prompt: str = "You are an autonomous ReAct agent.") -> None:
        self.llm_fn = llm_fn
        self.tools = tools
        self.tool_schemas = tool_schemas
        self.max_steps = max_steps
        self.max_concurrency = max_concurrency
        self.max_repeat_actions = max_repeat_actions
        self.tool_timeout = tool_timeout
        self.system_prompt = system_prompt
        self.semaphore = asyncio.Semaphore(max_concurrency)

    def _transition(self,new_state:AgentState,state_history:list[AgentState]) -> None:
        state_history.append(new_state)
    
    async def _llm_call(self,messages:list[dict[str,Any]],tool_schemas:list[dict[str,Any]]) -> dict[str,Any]:
        res = self.llm_fn(messages,tool_schemas)
        if inspect.isawaitable(res):
            return await res
        return res

    async def _execute_tool(self,name:str,args:dict[str,Any]) -> str:
        async with self.semaphore:
            if name not in self.tools:
                return f"Tool {name} not found!"
            try:
                fn = self.tools[name]
                if inspect.iscoroutinefunction(fn):
                    result = await asyncio.wait_for(fn(**args),self.tool_timeout)
                else:
                    result = await asyncio.wait_for(asyncio.to_thread(fn,**args),timeout=self.tool_timeout)
                return str(result)
            except TimeoutError:
                return f"Tool {name} execution timed out after {self.tool_timeout}s."
            except Exception as e:
                return f"Error executing {name}: {e}"

    async def run(self,query:str) -> AgentResult:
        state_history:list[str] = []
        steps:list[StepTrace] = []
        action_counts = {}
        messages = [{"role":"system","content":self.system_prompt},{"role":"user","content":query}]
        self._transition(AgentState.IDLE,state_history)
        for step_num in range(1,self.max_steps+1):
            self._transition(AgentState.THINKING,state_history)
            try:
                response = await self._llm_call(messages,self.tool_schemas or [])
            except Exception as e:
                self._transition(AgentState.FAILED,state_history)
                err_str = str(e)
                err_msg = f"Failed to generate response: {err_str}"
                if "429" in err_str or "quota" in err_str:
                    err_msg = "Quota exceeded! wait for sometime."
                return AgentResult("",AgentState.FAILED,state_history,steps,err_msg)
            thought = response.get("content") or ""
            tool_calls = response.get("tool_calls") or []
            if not tool_calls:
                self._transition(AgentState.COMPLETED,state_history)
                steps.append(StepTrace(step_num,thought,[],[]))
                return AgentResult(thought,AgentState.COMPLETED,state_history,steps,"completed")
            parsed_tools = []
            for tc in tool_calls:
                fn_info = tc.get("function",tc)
                name = fn_info.get("name")
                raw_args = fn_info.get("arguments") or {}
                args = raw_args if isinstance(raw_args,dict) else json.loads(raw_args)
                parsed_tools.append((name,args))
                tool_signature = f"{name}:{json.dumps(args, sort_keys=True)}"
                action_counts[tool_signature] = action_counts.get(tool_signature,0) + 1
                if action_counts[tool_signature] > self.max_repeat_actions:
                    self._transition(AgentState.FAILED,state_history)
                    return AgentResult("",AgentState.FAILED,state_history,steps,"cycle_detected")
            
            self._transition(AgentState.ACTING,state_history)
            tasks = [self._execute_tool(name,args) for name,args in parsed_tools]
            observations = list(await asyncio.gather(*tasks))
            
            self._transition(AgentState.OBSERVING,state_history)
            steps.append(StepTrace(step_num,thought,tool_calls,observations))
            messages.append({
                "role":"assistant",
                "content":thought or "",
                "tool_calls":tool_calls
            })
            for tc,observation in zip(tool_calls,observations):
                messages.append({
                    "role": "tool",
                    "name": tc["function"].get("name"),
                    "tool_call_id": tc.get("id") or "",
                    "content": observation
                })
            if step_num == self.max_steps:
                final_result = await self._llm_call(messages,[])
                final_answer = final_result.get("content") or ""
                self._transition(AgentState.FAILED,state_history)
                return AgentResult(final_answer,AgentState.FAILED,state_history,steps,"Max steps reached!")

async def main():
    tool_registry = ToolRegistry()
    tool_registry.register(get_git_status,"git_status")
    tool_registry.register(get_git_diff,"git_diff")
    tool_schemas = tool_registry.get_schemas()
    tools = tool_registry._tools
    agent = AsyncReACTAgent(call,tools,tool_schemas)
    response = await agent.run("generate a commit message based on my git changes")
    print(response)
if __name__ == "__main__":
    asyncio.run(main())