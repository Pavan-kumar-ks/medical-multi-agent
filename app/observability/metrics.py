"""In-process metrics collector for MedOrchestrator.

Tracks per-LLM-call and per-agent statistics:
  • token usage  (prompt + completion)
  • latency      (milliseconds)
  • error rates  (per agent + global)

Singleton — import `collector` anywhere and call `collector.record_llm_call(...)`.
Call `collector.summary()` at the end of a run to get a full report dict.
Call `collector.reset()` to clear between independent runs / tests.
"""
import time
import threading
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_METRICS_FILE = _PROJECT_ROOT / "logs" / "metrics.jsonl"
_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)


class MetricsCollector:
    """Thread-safe singleton metrics collector."""

    _instance: Optional["MetricsCollector"] = None
    _class_lock = threading.Lock()

    def __new__(cls) -> "MetricsCollector":
        with cls._class_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._lock = threading.Lock()
                inst._reset_state()
                cls._instance = inst
        return cls._instance

    # ── Internal state ────────────────────────────────────────────────────

    def _reset_state(self):
        self._calls: List[Dict[str, Any]] = []
        self._agent_calls:  Dict[str, int] = defaultdict(int)
        self._agent_errors: Dict[str, int] = defaultdict(int)
        self._agent_tokens: Dict[str, int] = defaultdict(int)
        self._agent_latency: Dict[str, List[float]] = defaultdict(list)
        self._session_start = time.time()

    # ── Public API ────────────────────────────────────────────────────────

    def record_llm_call(
        self,
        *,
        agent:       str,
        model:       str,
        tokens_in:   int,
        tokens_out:  int,
        latency_ms:  float,
        success:     bool,
        error:       Optional[str] = None,
        node:        str = "",
    ) -> None:
        """Record a single LLM call."""
        record = {
            "ts":          datetime.now(timezone.utc).isoformat(),
            "agent":       agent,
            "node":        node,
            "model":       model,
            "tokens_in":   tokens_in,
            "tokens_out":  tokens_out,
            "tokens_total": tokens_in + tokens_out,
            "latency_ms":  round(latency_ms, 2),
            "success":     success,
            "error":       error,
        }
        with self._lock:
            self._calls.append(record)
            self._agent_calls[agent]  += 1
            self._agent_tokens[agent] += tokens_in + tokens_out
            self._agent_latency[agent].append(latency_ms)
            if not success:
                self._agent_errors[agent] += 1

        # Append to JSONL metrics file
        try:
            with open(_METRICS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def record_agent_error(self, agent: str, error: str, node: str = "") -> None:
        """Record a non-LLM agent failure (e.g. tool crash, validation error)."""
        with self._lock:
            self._agent_errors[agent] += 1
        try:
            with open(_METRICS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts":    datetime.now(timezone.utc).isoformat(),
                    "type":  "agent_error",
                    "agent": agent,
                    "node":  node,
                    "error": error,
                }) + "\n")
        except Exception:
            pass

    def summary(self) -> Dict[str, Any]:
        """Return a full metrics summary dict."""
        with self._lock:
            calls = list(self._calls)
            a_calls   = dict(self._agent_calls)
            a_errors  = dict(self._agent_errors)
            a_tokens  = dict(self._agent_tokens)
            a_latency = {k: list(v) for k, v in self._agent_latency.items()}

        total     = len(calls)
        if not total:
            return {"total_llm_calls": 0}

        successful = [c for c in calls if c["success"]]
        failed     = [c for c in calls if not c["success"]]
        latencies  = [c["latency_ms"] for c in successful]

        per_agent: Dict[str, Any] = {}
        for agent, n in a_calls.items():
            lats = a_latency.get(agent, [])
            errs = a_errors.get(agent, 0)
            per_agent[agent] = {
                "calls":          n,
                "errors":         errs,
                "error_rate_pct": round(100 * errs / n, 1) if n else 0,
                "total_tokens":   a_tokens.get(agent, 0),
                "avg_latency_ms": round(sum(lats) / len(lats), 1) if lats else 0,
                "max_latency_ms": round(max(lats), 1) if lats else 0,
            }

        return {
            "session_duration_s":  round(time.time() - self._session_start, 1),
            "total_llm_calls":     total,
            "successful_calls":    len(successful),
            "failed_calls":        len(failed),
            "global_error_rate_pct": round(100 * len(failed) / total, 1),
            "total_tokens_in":     sum(c["tokens_in"]  for c in calls),
            "total_tokens_out":    sum(c["tokens_out"] for c in calls),
            "total_tokens":        sum(c["tokens_total"] for c in calls),
            "avg_latency_ms":      round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "p95_latency_ms":      round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if len(latencies) >= 20 else None,
            "max_latency_ms":      round(max(latencies), 1) if latencies else 0,
            "per_agent":           per_agent,
        }

    def print_summary(self) -> None:
        """Pretty-print the metrics summary to stdout."""
        s = self.summary()
        if not s.get("total_llm_calls"):
            print("No LLM calls recorded.")
            return
        W = 56
        print("\n" + "═" * W)
        print("  📊  SESSION METRICS")
        print("─" * W)
        print(f"  Duration        : {s['session_duration_s']}s")
        print(f"  LLM calls       : {s['total_llm_calls']}  "
              f"(✅ {s['successful_calls']}  ❌ {s['failed_calls']})")
        print(f"  Error rate      : {s['global_error_rate_pct']}%")
        print(f"  Avg latency     : {s['avg_latency_ms']} ms")
        print(f"  Max latency     : {s['max_latency_ms']} ms")
        print(f"  Total tokens    : {s['total_tokens']:,}  "
              f"(in: {s['total_tokens_in']:,}  out: {s['total_tokens_out']:,})")
        if s.get("per_agent"):
            print("─" * W)
            print("  Per-agent breakdown:")
            for agent, d in s["per_agent"].items():
                short = agent.replace("medorchestrator.", "")
                print(f"    {short:<28} {d['calls']:>3} calls  "
                      f"{d['avg_latency_ms']:>7} ms avg  "
                      f"{d['total_tokens']:>6} tok  "
                      f"err:{d['error_rate_pct']}%")
        print("═" * W + "\n")

    def reset(self) -> None:
        with self._lock:
            self._reset_state()


# ── Module-level singleton ────────────────────────────────────────────────────
collector = MetricsCollector()
