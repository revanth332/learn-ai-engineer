# Dynamic Tool Calling & Runtime Execution Engine

## 1. Executive Summary & Purpose

Modern LLM-powered applications do not just generate text—they interact with databases, REST APIs, calculators, code interpreters, and file systems. This capability is called **Tool Calling** (or Function Calling).

Instead of relying on heavy black-box frameworks (like LangChain or CrewAI), a production-grade tool engine is built on four core mechanical pillars:
1. **Dynamic Schema Generation**: Inspecting standard Python functions (type hints and docstrings) via reflection (`inspect` module) to generate strict JSON Schema definitions compliant with OpenAI / Anthropic tool calling formats.
2. **Tool Registry**: A central catalogue registering callable functions, validating uniqueness, and mapping schemas to executable callables.
3. **Safe Parameter Validation & Execution**: Validating incoming LLM-generated arguments against type signatures before execution, catching runtime exceptions cleanly without crashing the host process.
4. **Self-Healing Error Feedback Loops**: When an LLM passes malformed arguments or an exception occurs, formatting the exact stack trace and parameter error into a structured tool-result message so the LLM can self-correct on the next iteration.

---

## 2. Terminology & Core Definitions

| Term | Technical Definition |
| :--- | :--- |
| **Tool Calling / Function Calling** | A protocol where an LLM emits a structured JSON payload requesting to execute a specific external function with named arguments rather than generating unstructured natural language. |
| **JSON Schema** | A declarative vocabulary that annotates and validates the types, required fields, and constraints of JSON documents. |
| **Reflection / Introspection** | The runtime ability of Python to examine function signatures, parameter names, default values, and type annotations via `inspect.signature` and `typing.get_type_hints`. |
| **Docstring Parsing** | Extracting parameter descriptions from formatted Python docstrings (e.g. Google-style or Sphinx-style) to populate schema documentation for the model. |
| **Execution Sandboxing** | Isolating tool invocation so that errors (e.g. `ZeroDivisionError`, `KeyError`, `TimeoutError`) are trapped, formatted into descriptive error messages, and returned as structured output rather than halting the runtime. |
| **Self-Correction Feedback Loop** | Supplying tool execution error logs back to the LLM's conversation history in a subsequent turn, prompting the model to repair its arguments. |

---

## 3. High-Level Architecture

```
                       ┌─────────────────────────┐
                       │  Python Functions       │
                       │  (def get_weather(...)) │
                       └────────────┬────────────┘
                                    │ Decorator / Registration
                                    ▼
                       ┌─────────────────────────┐
                       │      ToolRegistry       │
                       │  - Stores callables     │
                       │  - Generates JSONSchema │
                       └────────────┬────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        [ Exported Tools Schema ]             [ Tool Execution Call ]
        (Sent to LLM in API prompt)           (Received from LLM response)
                  │                                   │
                  ▼                                   ▼
        ┌───────────────────┐               ┌───────────────────┐
        │ LLM API Provider  │               │ Parameter Validator│
        └─────────┬─────────┘               └─────────┬─────────┘
                  │                                   │
                  ▼ (LLM decides to call tool)         ▼
        { "name": "get_weather",             ┌───────────────────┐
          "arguments": "{\"city\": 42}" }    │ Sandbox Execution │
                  │                          └─────────┬─────────┘
                  │                                   │
                  └───────────────────────────────────┤
                                                      │
                       ┌──────────────────────────────┴──────────────┐
                       ▼                                             ▼
             [ Success Outcome ]                            [ Failure Outcome ]
             Returns: "Sunny, 22°C"                 Returns: "TypeError: 'city' must be str"
                       │                                             │
                       └──────────────────────┬──────────────────────┘
                                              ▼
                                 [ Structured Tool Message ]
                                 (Appended to Chat History)
```

---

## 4. JSON Schema Conversion Mechanics

When converting a Python function to an LLM tool schema:
- `str` -> `{"type": "string"}`
- `int` -> `{"type": "integer"}`
- `float` -> `{"type": "number"}`
- `bool` -> `{"type": "boolean"}`
- `list[T]` -> `{"type": "array", "items": {...}}`
- `dict` -> `{"type": "object"}`
- Parameters without default values are appended to the `"required"` array.

---

## 5. Next Implementation Deliverables

In `src/agent/tools.py`:
1. `python_type_to_json_type(py_type)`: Deterministic type mapping supporting primitives and container types.
2. `function_to_tool_schema(func)`: Dynamic introspection of callable signatures and docstrings into standard tool specs.
3. `ToolRegistry`: Class allowing registration (`@registry.register` or `register(func)`), schema export, and safe execution with structured error trapping.
