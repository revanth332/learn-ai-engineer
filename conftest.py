"""Pytest root configuration and automatic dashboard synchronization hook."""

import os
import sys
from pathlib import Path

# Ensure root directory is discoverable for imports
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

_session_results = {}


def pytest_runtest_logreport(report):
    """Capture individual test execution outcomes per test node."""
    if report.failed:
        nodeid = report.nodeid.replace("\\", "/")
        _session_results[nodeid] = "FAILED"
    elif report.when == "call" and report.passed:
        nodeid = report.nodeid.replace("\\", "/")
        _session_results[nodeid] = "PASSED"


def pytest_sessionfinish(session, exitstatus):
    """Automatically persist test results and regenerate dashboard.html upon session finish."""
    if os.environ.get("_DASHBOARD_UPDATING") == "1":
        return

    if not _session_results:
        return

    try:
        from generate_dashboard import (
            generate_html,
            load_cached_test_stats,
            save_cached_test_stats,
        )

        cache = load_cached_test_stats()
        cache.update(_session_results)
        save_cached_test_stats(cache)
        generate_html()
        print("\n[dashboard] Automatically synchronized dashboard.html with test execution.")
    except Exception as e:
        # Failsafe: never break test execution if dashboard synchronization encounters an issue
        print(f"\n[dashboard warning] Auto-sync skipped: {e}")
