"""CLI runner for the MedOrchestrator evaluation harness.

Usage:
    python -m app.evaluation.runner                     # run all 50+ queries
    python -m app.evaluation.runner --category cardiac  # one category only
    python -m app.evaluation.runner --llm-judge         # enable LLM-as-judge
    python -m app.evaluation.runner --limit 10          # first N queries
    python -m app.evaluation.runner --output report.json

The runner uses a lightweight mock of the pipeline (LLM-only, no live maps/scraper)
so it can run without a full environment. Set MOCK_RUN=false to use the real graph.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

# ── Path fix ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.evaluation.test_queries import TEST_QUERIES
from app.evaluation.harness      import EvaluationHarness


# ── Mock run function (no live API calls to maps/scraper) ─────────────────────

def _mock_run_fn(query: str) -> str:
    """Run the query through the LLM directly (no graph, no maps).

    Used for fast evaluation without a full environment.
    Calls llm_call with a fixed medical-assistant prompt.
    """
    try:
        from app.config import llm_call
    except ImportError:
        return f"[mock] No LLM available for query: {query}"

    prompt = f"""You are a medical assistant. A patient says:
"{query}"

Provide a brief clinical assessment:
1. Most likely diagnosis with confidence (%)
2. Whether this is an emergency (yes/no)
3. Recommended immediate action
4. One key test to confirm

Keep the answer under 150 words.
"""
    try:
        return llm_call(prompt, agent="eval_mock")
    except Exception as e:
        return f"[LLM error] {e}"


def _real_run_fn(query: str) -> str:
    """Run the query through the full LangGraph pipeline."""
    try:
        from app.orchestrator.graph    import build_graph
        from app.memory.vector_store   import load_vector_store
        from app.tools.formatter       import format_medical_response
        from app.orchestrator.agent_runner import reset_agent_trace

        load_vector_store()
        graph = build_graph()
        reset_agent_trace()

        result = graph.invoke({
            "user_input":     query,
            "chat_history":   [],
            "session_memory": {},
        })
        formatted = format_medical_response(result)
        return formatted.get("pretty_text", str(result))
    except Exception as e:
        return f"[graph error] {e}"


# ── Report printer ────────────────────────────────────────────────────────────

def _print_report(report: Dict[str, Any]) -> None:
    summary = report.get("summary", {})
    results = report.get("results", [])
    errors  = report.get("errors", [])
    W = 64

    print("\n" + "═" * W)
    print("  🧪  MEDORCHESTRATOR EVALUATION REPORT")
    print("─" * W)
    print(f"  Total queries  : {summary.get('total_queries', 0)}")
    print(f"  Passed (≥0.60) : {summary.get('passed', 0)}  "
          f"({summary.get('pass_rate_pct', 0)}%)")
    print(f"  Failed         : {summary.get('failed', 0)}")
    print(f"  Avg score      : {summary.get('avg_score', 0)}")
    print(f"  Min / Max      : {summary.get('min_score', 0)} / {summary.get('max_score', 0)}")
    print(f"  Hallucinations : {summary.get('hallucinations_flagged', 0)} queries flagged")
    print(f"  Errors         : {len(errors)}")

    # Per-category
    per_cat = summary.get("per_category", {})
    if per_cat:
        print("─" * W)
        print("  Per-category breakdown:")
        for cat, d in sorted(per_cat.items()):
            bar_fill = int(d["pass_rate"] * 10)
            bar = "█" * bar_fill + "░" * (10 - bar_fill)
            print(f"    {cat:<22} [{bar}] "
                  f"{int(d['pass_rate']*100):>3}%  avg:{d['avg_score']:.2f}  n={d['total']}")

    # Failed cases
    failed = [r for r in results if not r.get("passed")]
    if failed:
        print("─" * W)
        print("  ❌ Failed queries:")
        for r in failed[:10]:
            print(f"    [{r['query_id']:<14}] score={r['weighted_total']:.2f}  "
                  f"cat={r['category']}  hal={'Y' if r.get('hallucination_flagged') else 'N'}")
            if r.get("llm_reasoning"):
                print(f"          LLM judge: {r['llm_reasoning'][:80]}")

    # Hallucination flags
    hal_cases = [r for r in results if r.get("hallucination_flagged")]
    if hal_cases:
        print("─" * W)
        print("  ⚠️  Hallucination flags:")
        for r in hal_cases:
            for flag in r.get("hallucination_flags", []):
                print(f"    [{r['query_id']:<14}] {flag}")

    print("═" * W + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the MedOrchestrator evaluation harness."
    )
    parser.add_argument("--category",   type=str,  default=None,
                        help="Filter to a single category (e.g. cardiac, respiratory)")
    parser.add_argument("--limit",      type=int,  default=None,
                        help="Max number of queries to run")
    parser.add_argument("--llm-judge",  action="store_true",
                        help="Enable LLM-as-judge scoring (costs tokens)")
    parser.add_argument("--real-graph", action="store_true",
                        help="Use full LangGraph pipeline instead of mock LLM")
    parser.add_argument("--output",     type=str,  default=None,
                        help="Save full report to this JSON file")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop on first query error")

    args = parser.parse_args()

    # ── Filter queries ────────────────────────────────────────────────────
    queries = TEST_QUERIES
    if args.category:
        queries = [q for q in queries if q["category"] == args.category]
        if not queries:
            print(f"No queries found for category '{args.category}'")
            print(f"Available: {sorted(set(q['category'] for q in TEST_QUERIES))}")
            sys.exit(1)
    if args.limit:
        queries = queries[:args.limit]

    run_fn: Callable[[str], str] = _real_run_fn if args.real_graph else _mock_run_fn

    # ── Banner ────────────────────────────────────────────────────────────
    print(f"\n🧪  Running {len(queries)} queries "
          f"({'real graph' if args.real_graph else 'mock LLM'}, "
          f"{'LLM judge ON' if args.llm_judge else 'rule-based only'})…")

    # ── Run harness ───────────────────────────────────────────────────────
    harness = EvaluationHarness(use_llm_judge=args.llm_judge)
    t0      = time.time()
    report  = harness.run_batch(queries, run_fn, stop_on_error=args.stop_on_error)
    elapsed = round(time.time() - t0, 1)

    report["meta"] = {
        "elapsed_s":    elapsed,
        "mode":         "real_graph" if args.real_graph else "mock_llm",
        "llm_judge":    args.llm_judge,
        "query_count":  len(queries),
    }

    # ── Print ─────────────────────────────────────────────────────────────
    _print_report(report)
    print(f"  ⏱  Completed in {elapsed}s")

    # ── Save ──────────────────────────────────────────────────────────────
    output_path = args.output or str(
        Path(__file__).parent.parent.parent / "logs" / "eval_report.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  💾  Full report saved to: {output_path}\n")


if __name__ == "__main__":
    main()
