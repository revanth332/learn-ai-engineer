"""Dynamic Tool Calling & Runtime Execution Engine.

Implements reflection-based JSON schema generation from Python functions,
tool registries, argument validation, and sandboxed execution with structured
error recovery feedback loops.
"""

from typing import Any, Callable,Optional,Union,List
import inspect
import json
import typing
import types


def python_type_to_json_type(py_type: Any) -> dict[str, Any]:
    """Map Python types (str, int, float, bool, list[T], dict) to standard JSON Schema types.
    
    Args:
        py_type: A Python type or typing annotation (e.g. str, int, list[str], dict[str, Any]).
        
    Returns:
        Dictionary representing the JSON schema type (e.g. {"type": "string"}).
    """
    origin = typing.get_origin(py_type)
    args = typing.get_args(py_type)

    if origin in (typing.Union,getattr(types,"UnionType",None)):
        non_none_args = [t for t in args if t is not type(None)]
        if len(non_none_args) == 1:
            return python_type_to_json_type(non_none_args[0])
        return {"type":"string"}

    if origin is list or py_type is list:
        if args:
            return {"type":"array","items":python_type_to_json_type(args[0])}
        return {"type":"array"}
    
    PRIMITIVES = {
        str : {"type": "string"},
        int: {"type": "integer"},
        float : {"type": "number"},
        bool: {"type": "boolean"},
        dict: {"type": "object"},
    }

    return PRIMITIVES.get(py_type,{"type":"string"})


def function_to_tool_schema(func: Callable[..., Any], name_override: str | None = None) -> dict[str, Any]:
    """Introspect a Python callable to generate an OpenAI/Anthropic standard tool JSON Schema.
    
    Inspects:
    - Function name (or name_override).
    - Function docstring (for the overall tool description).
    - Parameter names, type annotations, and default values.
    - Required parameters (parameters without default values).
    
    Returns:
        A dictionary with format:
        {
            "type": "function",
            "function": {
                "name": "<func_name>",
                "description": "<docstring_summary>",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param_1": {"type": "..."}, ...
                    },
                    "required": ["param_1", ...]
                }
            }
        }
    """
    signature = inspect.signature(func)
    properties = {}
    req_params = []
    for name,param in signature.parameters.items():
        properties[name] = python_type_to_json_type(param.annotation)
        if param.default == inspect.Parameter.empty:
            req_params.append(name)
    return {
                "type": "function",
                "function": {
                    "name": name_override or func.__name__,
                    "description": inspect.getdoc(func) or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": req_params
                    }
                }
            }


class ToolExecutionResult:
    """Represents the structured outcome of a tool execution attempt."""

    def __init__(self, tool_name: str, success: bool, output: Any, error: str | None = None):
        self.tool_name = tool_name
        self.success = success
        self.output = output
        self.error = error

    def to_chat_message(self, tool_call_id: str | None = None) -> dict[str, Any]:
        """Convert the result into a standard LLM tool response message."""
        content = str(self.output) if self.success else f"Error executing tool '{self.tool_name}': {self.error}"
        msg: dict[str, Any] = {
            "role": "tool",
            "content": content,
            "name": self.tool_name,
        }
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        return msg


class ToolRegistry:
    """Registry managing callable tools, schema exports, and safe sandboxed execution."""

    def __init__(self):
        self._tools: dict[str, Callable[..., Any]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}

    def register(self, func: Callable[..., Any], name: str | None = None) -> Callable[..., Any]:
        """Register a callable tool function. Can be used as a decorator or direct method."""
        name = name or func.__name__
        tool_schema = function_to_tool_schema(func,name)
        self._schemas[name] = tool_schema
        self._tools[name] = func
        return func

    def get_tool(self, name: str) -> Callable[..., Any] | None:
        """Retrieve a registered callable by name."""
        return self._tools.get(name)

    def get_schemas(self) -> list[dict[str, Any]]:
        """Export all registered tool schemas formatted for LLM function calling prompts."""
        return list(self._schemas.values())

    def execute(self, tool_name: str, arguments: dict[str, Any] | str) -> ToolExecutionResult:
        """Safely execute a registered tool by name with parsed/validated arguments.
        
        Handles:
        - JSON string parsing if arguments is passed as raw JSON string.
        - Tool not found validation.
        - Runtime exception handling during execution without crashing host application.
        - Returning structured ToolExecutionResult.
        """
        result = None
        error = None
        success = False
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception as e:
                error = f"Invalid JSON arguments: {e}"
                return ToolExecutionResult(tool_name, success, result, error)
        if self._tools.get(tool_name, None):
            try:
                result = self._tools[tool_name](**arguments)
                success = True
            except Exception as e:
                error = f"Error executing tool '{tool_name}': {e}"
                result = None
        else:
            error = f"Error: Tool '{tool_name}' not found in registry."
        tool_result = ToolExecutionResult(tool_name, success, result, error)
        return tool_result
