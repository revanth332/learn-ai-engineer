"""Generates a standalone, interactive visual progress tracker dashboard.html."""

import json
import os
import subprocess
import time
from datetime import datetime


def get_test_stats() -> dict:
    """Run pytest with json-report or parse terminal output to get test metrics."""
    try:
        result = subprocess.run(
            ["python", "-m", "uv", "run", "python", "-m", "pytest", "--tb=no", "-v"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout + "\n" + result.stderr
    except Exception as e:
        output = str(e)

    # Count passes and fails per test file
    modules = {
        "tests/test_bpe_tokenizer.py": {
            "name": "Byte-Pair Encoding (BPE)",
            "tier": 1,
            "passed": 0,
            "failed": 0,
            "desc": "Byte-level BPE tokenization, vocabulary merging, and token encoding/decoding.",
        },
        "tests/test_context.py": {
            "name": "Context Budget & Framing",
            "tier": 1,
            "passed": 0,
            "failed": 0,
            "desc": "Token estimation, per-message framing overhead, and chronological context budget trimming.",
        },
        "tests/test_cost.py": {
            "name": "Token Cost Modeling",
            "tier": 1,
            "passed": 0,
            "failed": 0,
            "desc": "Model token-pricing and precise request/response cost calculation.",
        },
        "tests/test_streaming.py": {
            "name": "SSE Streaming Generator",
            "tier": 1,
            "passed": 0,
            "failed": 0,
            "desc": "Raw Server-Sent Events chunk reassembly, metadata filtering, and stream termination.",
        },
        "tests/test_structured_output.py": {
            "name": "Structured Outputs",
            "tier": 1,
            "passed": 0,
            "failed": 0,
            "desc": "Strict schema validation and deterministic Pydantic model parsing.",
        },
        "tests/test_json_repair.py": {
            "name": "JSON Repair & Regex Slicing",
            "tier": 1,
            "passed": 0,
            "failed": 0,
            "desc": "Non-hallucinating JSON repair (markdown fence stripping and trailing comma removal).",
        },
        "tests/test_resilience.py": {
            "name": "Rate Limiting & Backoff Jitter",
            "tier": 1,
            "passed": 0,
            "failed": 0,
            "desc": "Token Bucket rate limiter and Full Jitter Exponential Backoff.",
        },
        "tests/test_vector_math.py": {
            "name": "Vector Geometry & Cosine Math",
            "tier": 1,
            "passed": 0,
            "failed": 0,
            "desc": "Dot product, L2 norm, unit normalization, Euclidean distance, and cosine similarity.",
        },
        "tests/test_chunking.py": {
            "name": "Sliding-Window Text Chunking",
            "tier": 1,
            "passed": 0,
            "failed": 0,
            "desc": "Fixed-size character chunking with configurable overlap preserving semantic borders.",
        },
        "tests/test_vector_store.py": {
            "name": "In-Memory Vector Store",
            "tier": 1,
            "passed": 0,
            "failed": 0,
            "desc": "Exact nearest-neighbor in-memory vector database with top-k ranking and thresholding.",
        },
        "tests/test_bm25.py": {
            "name": "Okapi BM25 Keyword Search",
            "tier": 2,
            "passed": 0,
            "failed": 0,
            "desc": "Robertson-Spärck Jones IDF, term saturation (k1), and document length normalization (b).",
        },
        "tests/test_hybrid.py": {
            "name": "Reciprocal Rank Fusion (Hybrid)",
            "tier": 2,
            "passed": 0,
            "failed": 0,
            "desc": "Dual-engine hybrid search fusing Dense Vectors and BM25 rankings via RRF (k=60).",
        },
        "tests/test_reranking.py": {
            "name": "Cross-Encoder Re-Ranking",
            "tier": 2,
            "passed": 0,
            "failed": 0,
            "desc": "Stage-2 Cross-Encoder re-ranking with coverage, n-gram bigram matching, and proximity density.",
        },
        "tests/test_query_transform.py": {
            "name": "Query Transformation & HyDE",
            "tier": 2,
            "passed": 0,
            "failed": 0,
            "desc": "Deterministic synonym expansion, multi-query decomposition, HyDE prompt templates, and query fusion.",
        },
    }

    for line in output.splitlines():
        for test_file in modules:
            if test_file in line:
                if "PASSED" in line:
                    modules[test_file]["passed"] += 1
                elif "FAILED" in line:
                    modules[test_file]["failed"] += 1

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


def get_codebase_stats() -> dict:
    """Calculate source lines of code vs test lines of code."""
    src_lines = 0
    test_lines = 0

    for root, _, files in os.walk("src"):
        for f in files:
            if f.endswith(".py"):
                with open(os.path.join(root, f), "r", encoding="utf-8") as file:
                    src_lines += len(file.readlines())

    for root, _, files in os.walk("tests"):
        for f in files:
            if f.endswith(".py"):
                with open(os.path.join(root, f), "r", encoding="utf-8") as file:
                    test_lines += len(file.readlines())

    return {
        "src_lines": src_lines,
        "test_lines": test_lines,
        "ratio": round(test_lines / src_lines, 2) if src_lines > 0 else 0.0,
    }


def generate_html() -> None:
    """Generate self-contained dashboard.html."""
    stats = get_test_stats()
    code_stats = get_codebase_stats()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Tier calculations
    tier1_modules = [m for m in stats["modules"].values() if m["tier"] == 1]
    tier2_modules = [m for m in stats["modules"].values() if m["tier"] == 2]

    t1_passed = sum(m["passed"] for m in tier1_modules)
    t1_total = sum(m["passed"] + m["failed"] for m in tier1_modules)
    t1_pct = round((t1_passed / t1_total * 100), 1) if t1_total > 0 else 0.0

    t2_passed = sum(m["passed"] for m in tier2_modules)
    t2_total = sum(m["passed"] + m["failed"] for m in tier2_modules)
    t2_pct = round((t2_passed / t2_total * 100), 1) if t2_total > 0 else 0.0

    # Determine dynamic active module
    active_module = None
    for path, m in stats["modules"].items():
        if m["failed"] > 0 or (m["passed"] == 0 and m["failed"] == 0):
            active_module = m
            break

    if not active_module:
        active_name = "All Current Modules Verified!"
        active_tier = "Tier 2 Complete"
        active_desc = "All unit tests across current modules are passing at 100%."
    else:
        active_name = active_module["name"]
        active_tier = f"Tier {active_module['tier']}: Active Component"
        active_desc = active_module.get("desc", "Component currently under active implementation and test verification.")

    # Modules JSON for frontend rendering
    modules_json = json.dumps(stats["modules"])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Engineering Mastery Tracker</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <script>
        tailwind.config = {{
            darkMode: 'class',
            theme: {{
                extend: {{
                    colors: {{
                        brand: {{
                            50: '#f0fdfa',
                            100: '#ccfbf1',
                            500: '#0d9488',
                            600: '#0f766e',
                            900: '#134e4a',
                        }},
                    }}
                }}
            }}
        }}
    </script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        body {{
            font-family: 'Plus Jakarta Sans', sans-serif;
        }}
        code, .font-mono {{
            font-family: 'JetBrains Mono', monospace;
        }}
    </style>
</head>
<body class="min-h-screen bg-slate-50 text-slate-800 p-4 md:p-8 antialiased selection:bg-teal-500/20 selection:text-teal-900">
    <div class="max-w-7xl mx-auto space-y-8">

        <!-- HEADER -->
        <header class="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-200 pb-6">
            <div>
                <div class="flex items-center gap-3">
                    <div class="p-2.5 rounded-xl bg-teal-50 border border-teal-200 text-teal-700 shadow-sm">
                        <i data-lucide="cpu" class="w-6 h-6"></i>
                    </div>
                    <div>
                        <h1 class="text-2xl md:text-3xl font-extrabold tracking-tight text-slate-900 flex items-center gap-2">
                            AI Engineer Progress Radar
                            <span class="text-xs px-2.5 py-0.5 rounded-full bg-teal-100 text-teal-800 border border-teal-300 font-mono font-semibold">FIRST-PRINCIPLES</span>
                        </h1>
                        <p class="text-sm text-slate-500">Autonomous First-Principles Mastery & Test Verification System</p>
                    </div>
                </div>
            </div>
            <div class="flex items-center gap-3 text-xs font-mono text-slate-600 bg-white px-4 py-2.5 rounded-xl border border-slate-200 shadow-sm">
                <i data-lucide="clock" class="w-4 h-4 text-teal-600"></i>
                <span>Last Synced: <span class="text-slate-900 font-bold">{now_str}</span></span>
            </div>
        </header>

        <!-- ACTIVE STANDING HERO BANNER (DYNAMIC) -->
        <section class="relative overflow-hidden rounded-2xl bg-gradient-to-br from-white via-white to-teal-50/70 border border-slate-200 p-6 md:p-8 shadow-sm">
            <div class="flex flex-col md:flex-row md:items-center justify-between gap-6">
                <div class="space-y-2">
                    <div class="flex items-center gap-2">
                        <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-amber-100 text-amber-800 border border-amber-300">
                            <span class="w-2 h-2 rounded-full bg-amber-500 animate-pulse"></span>
                            Current In-Flight Milestone
                        </span>
                        <span class="text-xs font-mono text-slate-500 font-medium">{active_tier}</span>
                    </div>
                    <h2 class="text-2xl md:text-3xl font-bold text-slate-900">{active_name}</h2>
                    <p class="text-sm text-slate-600 max-w-2xl leading-relaxed">
                        {active_desc}
                    </p>
                </div>
                <div class="flex flex-col gap-2 min-w-[220px] bg-slate-50 p-4 rounded-xl border border-slate-200 shadow-sm">
                    <div class="flex justify-between text-xs font-mono text-slate-600">
                        <span>Phase Pass Rate</span>
                        <span class="text-teal-700 font-bold">{stats['pass_rate']}%</span>
                    </div>
                    <div class="w-full bg-slate-200 h-2.5 rounded-full overflow-hidden">
                        <div class="bg-gradient-to-r from-teal-500 to-emerald-500 h-full rounded-full transition-all duration-500" style="width: {stats['pass_rate']}%"></div>
                    </div>
                    <div class="flex justify-between text-xs font-mono text-slate-500 pt-1">
                        <span class="font-semibold text-emerald-700">{stats['total_passed']} Passed</span>
                        <span class="font-semibold text-amber-700">{stats['total_failed']} Pending</span>
                    </div>
                </div>
            </div>
        </section>

        <!-- VITALS & KEY METRICS GRID -->
        <section class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            
            <!-- Card 1: Overall Tests -->
            <div class="bg-white border border-slate-200 rounded-xl p-5 hover:border-slate-300 shadow-sm transition">
                <div class="flex items-center justify-between text-slate-500 mb-3">
                    <span class="text-xs font-bold uppercase tracking-wider">Total Tests</span>
                    <i data-lucide="shield-check" class="w-5 h-5 text-emerald-600"></i>
                </div>
                <div class="flex items-baseline gap-2">
                    <span class="text-3xl font-extrabold text-slate-900 font-mono">{stats['total_passed']}</span>
                    <span class="text-sm text-slate-500 font-mono">/ {stats['total_tests']}</span>
                </div>
                <p class="text-xs text-emerald-700 mt-2 flex items-center gap-1 font-mono font-medium">
                    <i data-lucide="check" class="w-3.5 h-3.5"></i> {stats['pass_rate']}% verified
                </p>
            </div>

            <!-- Card 2: Tier 1 Foundations -->
            <div class="bg-white border border-slate-200 rounded-xl p-5 hover:border-slate-300 shadow-sm transition">
                <div class="flex items-center justify-between text-slate-500 mb-3">
                    <span class="text-xs font-bold uppercase tracking-wider">Tier 1: Foundations</span>
                    <i data-lucide="award" class="w-5 h-5 text-teal-600"></i>
                </div>
                <div class="flex items-baseline gap-2">
                    <span class="text-3xl font-extrabold text-teal-600 font-mono">{t1_pct}%</span>
                    <span class="text-xs text-slate-500 font-mono">({t1_passed}/{t1_total})</span>
                </div>
                <p class="text-xs text-slate-600 mt-2 flex items-center gap-1 font-medium">
                    <span class="w-2 h-2 rounded-full bg-teal-500"></span> 10 of 10 Modules Complete
                </p>
            </div>

            <!-- Card 3: Tier 2 Intermediate -->
            <div class="bg-white border border-slate-200 rounded-xl p-5 hover:border-slate-300 shadow-sm transition">
                <div class="flex items-center justify-between text-slate-500 mb-3">
                    <span class="text-xs font-bold uppercase tracking-wider">Tier 2: Intermediate</span>
                    <i data-lucide="git-merge" class="w-5 h-5 text-amber-600"></i>
                </div>
                <div class="flex items-baseline gap-2">
                    <span class="text-3xl font-extrabold text-amber-600 font-mono">{t2_pct}%</span>
                    <span class="text-xs text-slate-500 font-mono">({t2_passed}/{t2_total})</span>
                </div>
                <p class="text-xs text-amber-700 mt-2 flex items-center gap-1 font-medium">
                    <span class="w-2 h-2 rounded-full bg-amber-500 animate-pulse"></span> Active: {active_name}
                </p>
            </div>

            <!-- Card 4: Code vs Test Ratio -->
            <div class="bg-white border border-slate-200 rounded-xl p-5 hover:border-slate-300 shadow-sm transition">
                <div class="flex items-center justify-between text-slate-500 mb-3">
                    <span class="text-xs font-bold uppercase tracking-wider">Test-to-Code Ratio</span>
                    <i data-lucide="code-2" class="w-5 h-5 text-indigo-600"></i>
                </div>
                <div class="flex items-baseline gap-2">
                    <span class="text-3xl font-extrabold text-slate-900 font-mono">{code_stats['ratio']}x</span>
                    <span class="text-xs text-slate-500 font-mono">({code_stats['test_lines']} test / {code_stats['src_lines']} src)</span>
                </div>
                <p class="text-xs text-indigo-600 mt-2 font-mono font-medium">
                    Production test harness
                </p>
            </div>

        </section>

        <!-- ROADMAP TIERS & MODULE BREAKDOWN WITH DEBOUNCED SEARCH -->
        <section class="space-y-4">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <h3 class="text-lg font-bold text-slate-900 flex items-center gap-2">
                    <i data-lucide="layers" class="w-5 h-5 text-teal-600"></i>
                    Component Verification Matrix
                </h3>
                
                <!-- Debounced Search Filter -->
                <div class="relative w-full sm:w-72">
                    <i data-lucide="search" class="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2"></i>
                    <input 
                        type="text" 
                        id="searchInput" 
                        placeholder="Search components or tests..." 
                        class="w-full bg-white border border-slate-300 rounded-lg pl-9 pr-4 py-2 text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 transition font-mono shadow-sm"
                    />
                </div>
            </div>

            <!-- MODULE CARDS GRID -->
            <div id="moduleGrid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <!-- Injected via JavaScript -->
            </div>
        </section>

        <!-- CONCEPT VAULT -->
        <section class="bg-white border border-slate-200 rounded-2xl p-6 space-y-4 shadow-sm">
            <h3 class="text-lg font-bold text-slate-900 flex items-center gap-2">
                <i data-lucide="brain" class="w-5 h-5 text-teal-600"></i>
                Core Algorithms & Mathematical Invariants Mastered
            </h3>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
                <div class="p-3.5 rounded-lg bg-slate-50 border border-slate-200 space-y-1">
                    <div class="font-bold text-teal-800 font-mono">Byte-Pair Encoding</div>
                    <p class="text-slate-600">Greedy pair frequency counting, 256 base byte vocab, UTF-8 lossless decoding.</p>
                </div>
                <div class="p-3.5 rounded-lg bg-slate-50 border border-slate-200 space-y-1">
                    <div class="font-bold text-teal-800 font-mono">Token Bucket Limiter</div>
                    <p class="text-slate-600">Drift-safe monotonic clock, continuous token refill math, burst capacity protection.</p>
                </div>
                <div class="p-3.5 rounded-lg bg-slate-50 border border-slate-200 space-y-1">
                    <div class="font-bold text-teal-800 font-mono">Full Jitter Backoff</div>
                    <p class="text-slate-600">min(max_delay, base * 2^attempt) * uniform(0, 1) to eliminate thundering herds.</p>
                </div>
                <div class="p-3.5 rounded-lg bg-slate-50 border border-slate-200 space-y-1">
                    <div class="font-bold text-teal-800 font-mono">Cosine & Euclidean Math</div>
                    <p class="text-slate-600">Exact L2 unit normalization, dot product speedup, cosine similarity [-1.0, 1.0].</p>
                </div>
                <div class="p-3.5 rounded-lg bg-slate-50 border border-slate-200 space-y-1">
                    <div class="font-bold text-teal-800 font-mono">Fixed-Size Chunking</div>
                    <p class="text-slate-600">Sliding character window with configurable overlap preserving semantic borders.</p>
                </div>
                <div class="p-3.5 rounded-lg bg-slate-50 border border-slate-200 space-y-1">
                    <div class="font-bold text-teal-800 font-mono">In-Memory Vector DB</div>
                    <p class="text-slate-600">Exact top-k nearest neighbor retrieval with dynamic dimensionality & thresholding.</p>
                </div>
                <div class="p-3.5 rounded-lg bg-slate-50 border border-slate-200 space-y-1">
                    <div class="font-bold text-teal-800 font-mono">Okapi BM25 Sparse</div>
                    <p class="text-slate-600">Term frequency saturation (k1), length normalization (b), and RSJ floor-smoothed IDF.</p>
                </div>
                <div class="p-3.5 rounded-lg bg-slate-50 border border-slate-200 space-y-1">
                    <div class="font-bold text-teal-800 font-mono">Reciprocal Rank Fusion</div>
                    <p class="text-slate-600">sum(1 / (60 + rank)) to unify disparate dense and sparse scores without distortion.</p>
                </div>
            </div>
        </section>

        <!-- FOOTER -->
        <footer class="pt-4 border-t border-slate-200 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 font-mono">
            <span>Raven AI Developer Coach Harness</span>
            <span>Run <code class="text-teal-700 bg-teal-50 px-1 py-0.5 rounded border border-teal-200 font-bold">python generate_dashboard.py</code> to refresh stats</span>
        </footer>

    </div>

    <!-- CLIENT SCRIPT: Modular Renderer & Debounced Search -->
    <script>
        const rawModules = {modules_json};

        // Debounce utility (ensures debouncing on all search inputs)
        function debounce(func, wait) {{
            let timeout;
            return function(...args) {{
                clearTimeout(timeout);
                timeout = setTimeout(() => func.apply(this, args), wait);
            }};
        }}

        function renderModules(filterText = '') {{
            const grid = document.getElementById('moduleGrid');
            grid.innerHTML = '';
            const query = filterText.toLowerCase().trim();

            Object.entries(rawModules).forEach(([path, mod]) => {{
                const matches = mod.name.toLowerCase().includes(query) || path.toLowerCase().includes(query);
                if (!matches) return;

                const isComplete = mod.failed === 0 && mod.passed > 0;
                const badgeClass = isComplete 
                    ? 'bg-emerald-100 text-emerald-800 border-emerald-300' 
                    : (mod.passed > 0 ? 'bg-amber-100 text-amber-800 border-amber-300' : 'bg-slate-100 text-slate-700 border-slate-300');
                const badgeText = isComplete ? 'PASSED' : (mod.failed > 0 ? 'IN PROGRESS' : 'PENDING');

                const card = document.createElement('div');
                card.className = 'p-4 rounded-xl bg-white border border-slate-200 hover:border-slate-300 shadow-sm transition flex flex-col justify-between space-y-3';

                card.innerHTML = `
                    <div>
                        <div class="flex items-center justify-between gap-2 mb-1.5">
                            <span class="text-xs px-2 py-0.5 rounded font-mono font-semibold uppercase border ${{badgeClass}}">
                                ${{badgeText}}
                            </span>
                            <span class="text-xs font-mono text-slate-500 font-medium">Tier ${{mod.tier}}</span>
                        </div>
                        <h4 class="text-sm font-bold text-slate-900">${{mod.name}}</h4>
                        <div class="text-xs font-mono text-slate-500 mt-1 truncate" title="${{path}}">${{path}}</div>
                    </div>
                    <div class="pt-2 border-t border-slate-100 flex items-center justify-between text-xs font-mono">
                        <span class="text-slate-500">Tests:</span>
                        <span class="text-slate-900 font-bold">${{mod.passed}} passed ${{mod.failed > 0 ? `<span class="text-rose-600 font-semibold">/ ${{mod.failed}} failed</span>` : ''}}</span>
                    </div>
                `;

                grid.appendChild(card);
            }});

            lucide.createIcons();
        }}

        // Setup Debounced Search Input
        const searchInput = document.getElementById('searchInput');
        const handleSearch = debounce((e) => {{
            renderModules(e.target.value);
        }}, 200);

        searchInput.addEventListener('input', handleSearch);

        // Initial Render
        renderModules();
        lucide.createIcons();
    </script>
</body>
</html>
"""
    with open("dashboard.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("Successfully generated dashboard.html!")


if __name__ == "__main__":
    generate_html()
