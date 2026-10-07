# Architectural Guide: Automated Test Synchronization & In-Process Dashboard Hooks

## 1. Executive Summary & Problem Space

In modern AI engineering workflows, engineers alternate between rapid Test-Driven Development (TDD) on focused sub-modules and high-level architectural inspections across the entire system. 

A common anti-pattern in developer tooling is **observability friction**: developers run tests in terminal sessions, but visual dashboards or reporting interfaces remain stale until someone manually executes a secondary build command or script. 

Furthermore, naively attempting to automate test reporting introduces three severe technical traps:
1. **The Partial-Run Obliteration Bug**: Running a single unit test (e.g., `pytest tests/test_query_transform.py`) only reports outcomes for that specific file. A naive reporting hook that overwrites global metrics will falsely report that all other passing test modules have vanished.
2. **The Subprocess Infinite-Loop Trap**: If a test hook spawns a subprocess running the test suite to recalculate metrics, it creates an infinite fork bomb of test runners triggering test runners.
3. **Execution Latency Penalty**: Adding hundreds of milliseconds of overhead to every test run degrades the developer feedback loop and breaks rapid iterative rhythm.

The solution is an **In-Process Upsert-Cached Pytest Hook Architecture**.

---

## 2. Key Terms & Architectural Definitions

| Term | Technical Definition |
| :--- | :--- |
| **Pytest Hook Specification** | A lifecycle callback in Pytest's plugin architecture (`pluggy`) that allows external logic to run during test collection, execution, and teardown without modifying test code. |
| **`pytest_runtest_logreport`** | A native Pytest hook that receives test report objects (`TestReport`) after each phase (`setup`, `call`, `teardown`) of every individual test function. |
| **`pytest_sessionfinish`** | A native hook invoked once at the conclusion of the entire test session, before Pytest prints summary output and exits. |
| **Test Node ID** | A unique canonical identifier for an exact test case in the format `path/to/test_file.py::test_function_name` (e.g., `tests/test_query_transform.py::test_decompose_query_conjunctions`). |
| **Upsert Caching** | An update-or-insert mutation strategy where existing keys are updated and new keys are appended, preventing partial operations from deleting unaffected historical state. |
| **Zero-Fork / In-Process Rendering** | Executing compilation and rendering routines directly inside the current Python interpreter process, avoiding `subprocess.Popen` / `subprocess.run` overhead. |
| **Recursion Guard Flag** | An environment variable (e.g., `_DASHBOARD_UPDATING=1`) used to detect if a test run is initiated by a build script, preventing recursive callback cycles. |

---

## 3. High-Level System Architecture

```
                       [ Developer Terminal ]
                                 │
                 Runs: pytest tests/test_xyz.py
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Pytest Test Runner    │
                     └───────────┬───────────┘
                                 │
     For each test case executed │ (phase = 'call' or failed 'setup')
                                 ▼
                     ┌───────────────────────┐
                     │ pytest_runtest_log    │
                     │  _report hook         │
                     └───────────┬───────────┘
                                 │ Collects Node ID -> State
                                 ▼
                     ┌───────────────────────┐
                     │ In-Memory Accumulator │
                     │ {_results[nodeid]}    │
                     └───────────┬───────────┘
                                 │
             Session completes   │ (pytest_sessionfinish hook)
                                 ▼
            ┌─────────────────────────────────────────┐
            │ Persistent Upsert Cache                 │
            │ (.pytest_cache/test_results.json)       │
            │   - Loads existing test outcomes        │
            │   - Merges current run's node IDs       │
            │   - Writes updated state back to disk   │
            └────────────────────┬────────────────────┘
                                 │
                                 ▼
            ┌─────────────────────────────────────────┐
            │ In-Process HTML Dashboard Generator     │
            │   - Aggregates module-level metrics     │
            │   - Evaluates curriculum & badge status │
            │   - Writes dashboard.html (<15ms)       │
            └─────────────────────────────────────────┘
```

---

## 4. Detailed Component Mechanics

### Component A: Granular Test Capture Hook

Pytest creates a `TestReport` instance whenever a test moves through a phase. To accurately register test status without recording duplicate passes during `setup` or `teardown`, we isolate the `call` phase, while catching critical setup failures (such as fixture exceptions):

```python
# conftest.py
_session_results = {}

def pytest_runtest_logreport(report):
    if report.failed:
        # Catches both execution failures and setup/fixture failures
        nodeid = report.nodeid.replace("\\", "/")
        _session_results[nodeid] = "FAILED"
    elif report.when == "call" and report.passed:
        nodeid = report.nodeid.replace("\\", "/")
        _session_results[nodeid] = "PASSED"
```

### Component B: Granular Upsert Cache

Using the canonical `nodeid` as the primary key guarantees that:
- Running `pytest tests/test_bpe_tokenizer.py` updates only BPE test cases.
- Running `pytest tests/test_query_transform.py::test_expand_query_synonyms_single_match` updates only that single function.
- All other passing test modules retain their status in the cache.

```python
# generate_dashboard.py
CACHE_FILE = os.path.join(".pytest_cache", "test_results.json")

def load_cached_test_stats() -> dict[str, str]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cached_test_stats(cache: dict[str, str]) -> None:
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
```

### Component C: Aggregation Pipeline

Raw node IDs are grouped by their file paths and mapped against the system's defined modules:

```python
def aggregate_module_stats(cache: dict[str, str]) -> dict:
    modules = {file: {**meta, "passed": 0, "failed": 0} for file, meta in MODULE_METADATA.items()}

    for nodeid, outcome in cache.items():
        file_part = nodeid.split("::")[0].replace("\\", "/")
        if file_part in modules:
            if outcome == "PASSED":
                modules[file_part]["passed"] += 1
            elif outcome == "FAILED":
                modules[file_part]["failed"] += 1

    total_passed = sum(m["passed"] for m in modules.values())
    total_failed = sum(m["failed"] for m in modules.values())
    total_tests = total_passed + total_failed

    return {
        "modules": modules,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_tests": total_tests,
        "pass_rate": round((total_passed / total_tests * 100), 1) if total_tests > 0 else 0.0,
    }
```

### Component D: In-Process Session Finalization

At session conclusion, the hook loads the cache, writes updates, and invokes dashboard regeneration directly in memory:

```python
def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("_DASHBOARD_UPDATING") == "1":
        return
    if not _session_results:
        return

    try:
        from generate_dashboard import load_cached_test_stats, save_cached_test_stats, generate_html
        cache = load_cached_test_stats()
        cache.update(_session_results)
        save_cached_test_stats(cache)
        generate_html()
        print("\n[dashboard] Automatically synchronized dashboard.html with test execution.")
    except Exception as e:
        # Never fail developer test runs if telemetry/dashboard sync fails
        print(f"\n[dashboard warning] Auto-sync skipped: {e}")
```

---

## 5. Real-World Engineering Applications

### 1. High-Speed Local TDD
When refactoring tokenizers or vector retrieval algorithms, an engineer runs unit tests hundreds of times a day. If documentation or tracking dashboards require manual re-generation, they quickly become outdated. Automated in-process hooks keep all artifacts synchronized with zero cognitive burden.

### 2. CI/CD Pipeline Artifact Generation
In production CI runners (e.g., GitHub Actions, GitLab CI), this pattern enables generating verified, standalone HTML report artifacts and metric badges during the normal `pytest` execution step without requiring separate reporting steps or third-party reporting containers.

### 3. Distributed Test Suites & Flaky Test Quarantine
By recording outcomes per `nodeid` with timestamps, the upsert cache pattern extends naturally into flaky test tracking: tests that alternate between `PASSED` and `FAILED` across successive runs can be automatically flagged in the telemetry layer for quarantine.

---

## 6. Implementation Checklist & Best Practices

1. **Place Hook in Project Root**: Ensure `conftest.py` sits at the repository root so it is discovered by Pytest regardless of which sub-directory or test module is invoked.
2. **Store Cache in Git-Ignored Location**: Use `.pytest_cache/` or `.cache/` so temporary test execution traces do not pollute git diffs or commit history.
3. **Fail-Safe Exception Isolation**: Always wrap telemetry and synchronization code in `try...except` blocks within `pytest_sessionfinish`. A dashboard generation error must never cause a clean test suite run to return a non-zero exit code.
4. **Guard Against Subprocess Recursion**: Always pass an environment variable guard flag (e.g., `_DASHBOARD_UPDATING=1`) if any tool ever invokes the test runner programmatically.
