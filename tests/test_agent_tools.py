"""Tests for the Dynamic Tool Calling & Runtime Execution Engine."""

import json
from typing import Any, Optional

import pytest

from src.agent.tools import (
    ToolExecutionResult,
    ToolRegistry,
    function_to_tool_schema,
    python_type_to_json_type,
)


class TestPythonTypeToJsonType:
    def test_primitive_types(self):
        assert python_type_to_json_type(str) == {"type": "string"}
        assert python_type_to_json_type(int) == {"type": "integer"}
        assert python_type_to_json_type(float) == {"type": "number"}
        assert python_type_to_json_type(bool) == {"type": "boolean"}
        assert python_type_to_json_type(dict) == {"type": "object"}

    def test_generic_list_types(self):
        assert python_type_to_json_type(list) == {"type": "array"}
        assert python_type_to_json_type(list[str]) == {
            "type": "array",
            "items": {"type": "string"},
        }
        assert python_type_to_json_type(list[int]) == {
            "type": "array",
            "items": {"type": "integer"},
        }
        assert python_type_to_json_type(list[float]) == {
            "type": "array",
            "items": {"type": "number"},
        }

    def test_optional_and_union_types(self):
        assert python_type_to_json_type(str | None) == {"type": "string"}
        assert python_type_to_json_type(Optional[int]) == {"type": "integer"}

    def test_fallback_for_unknown_or_any(self):
        assert python_type_to_json_type(Any) == {"type": "string"}
        assert python_type_to_json_type(object) == {"type": "string"}


class TestFunctionToToolSchema:
    def test_basic_function_schema(self):
        def calculate_tax(income: float, state: str) -> float:
            """Calculate state income tax for an individual."""
            return income * 0.05

        schema = function_to_tool_schema(calculate_tax)

        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "calculate_tax"
        assert "Calculate state income tax" in fn["description"]
        assert fn["parameters"]["type"] == "object"
        assert "income" in fn["parameters"]["properties"]
        assert fn["parameters"]["properties"]["income"] == {"type": "number"}
        assert "state" in fn["parameters"]["properties"]
        assert fn["parameters"]["properties"]["state"] == {"type": "string"}
        assert sorted(fn["parameters"]["required"]) == ["income", "state"]

    def test_function_with_default_arguments(self):
        def fetch_logs(service: str, limit: int = 100, verbose: bool = False) -> list[str]:
            """Retrieve service logs with optional pagination."""
            return []

        schema = function_to_tool_schema(fetch_logs)
        fn = schema["function"]

        assert fn["name"] == "fetch_logs"
        assert fn["parameters"]["required"] == ["service"]
        assert fn["parameters"]["properties"]["service"] == {"type": "string"}
        assert fn["parameters"]["properties"]["limit"] == {"type": "integer"}
        assert fn["parameters"]["properties"]["verbose"] == {"type": "boolean"}

    def test_function_with_name_override(self):
        def internal_query(q: str) -> str:
            """Execute raw search query."""
            return q

        schema = function_to_tool_schema(internal_query, name_override="search_database")
        assert schema["function"]["name"] == "search_database"

    def test_function_without_docstring(self):
        def no_doc(x: int) -> int:
            return x * 2

        schema = function_to_tool_schema(no_doc)
        assert schema["function"]["name"] == "no_doc"
        assert schema["function"]["description"] == ""
        assert schema["function"]["parameters"]["required"] == ["x"]


class TestToolExecutionResult:
    def test_success_message_formatting(self):
        result = ToolExecutionResult(tool_name="get_weather", success=True, output="72F Sunny")
        msg = result.to_chat_message(tool_call_id="call_abc123")

        assert msg["role"] == "tool"
        assert msg["name"] == "get_weather"
        assert msg["content"] == "72F Sunny"
        assert msg["tool_call_id"] == "call_abc123"

    def test_failure_message_formatting(self):
        result = ToolExecutionResult(
            tool_name="divide", success=False, output=None, error="division by zero"
        )
        msg = result.to_chat_message()

        assert msg["role"] == "tool"
        assert msg["name"] == "divide"
        assert "Error executing tool 'divide'" in msg["content"]
        assert "division by zero" in msg["content"]
        assert "tool_call_id" not in msg


class TestToolRegistry:
    def test_register_and_get_schemas(self):
        registry = ToolRegistry()

        @registry.register
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        def multiply(x: float, y: float) -> float:
            """Multiply two numbers."""
            return x * y

        registry.register(multiply, name="custom_mult")

        assert registry.get_tool("add") is not None
        assert registry.get_tool("custom_mult") is not None
        assert registry.get_tool("multiply") is None

        schemas = registry.get_schemas()
        assert len(schemas) == 2
        names = {s["function"]["name"] for s in schemas}
        assert names == {"add", "custom_mult"}

    def test_successful_execution_with_dict_and_json_string(self):
        registry = ToolRegistry()

        @registry.register
        def compute_power(base: int, exp: int = 2) -> int:
            """Compute base raised to exp."""
            return base**exp

        # Dict execution
        res_dict = registry.execute("compute_power", {"base": 3, "exp": 3})
        assert res_dict.success is True
        assert res_dict.output == 27
        assert res_dict.error is None

        # JSON string execution
        res_json = registry.execute("compute_power", '{"base": 4, "exp": 2}')
        assert res_json.success is True
        assert res_json.output == 16

    def test_execution_tool_not_found(self):
        registry = ToolRegistry()
        res = registry.execute("non_existent_tool", {})

        assert res.success is False
        assert "not found" in res.error.lower()

    def test_execution_invalid_json_string(self):
        registry = ToolRegistry()

        @registry.register
        def ping() -> str:
            """Ping check."""
            return "pong"

        res = registry.execute("ping", "{malformed_json:")
        assert res.success is False
        assert "json" in res.error.lower()

    def test_execution_sandboxed_exception_handling(self):
        registry = ToolRegistry()

        @registry.register
        def safe_divide(a: float, b: float) -> float:
            """Divide a by b."""
            return a / b

        res = registry.execute("safe_divide", {"a": 10.0, "b": 0.0})
        assert res.success is False
        assert res.output is None
        assert "division by zero" in res.error.lower() or "zerodivisionerror" in res.error.lower()

    def test_execution_missing_required_arguments(self):
        registry = ToolRegistry()

        @registry.register
        def greet(first_name: str, last_name: str) -> str:
            """Greet a person."""
            return f"Hello, {first_name} {last_name}"

        res = registry.execute("greet", {"first_name": "Ada"})
        assert res.success is False
        assert "missing" in res.error.lower() or "typeerror" in res.error.lower()
