"""Safe agent invocation helper with structured logging and observability.

call_agent() wraps any agent function with:
  • Try/except capturing all exceptions
  • Configurable retries with exponential backoff
  • Optional fallback value / callable returned on failure
  • Standardised return shape: {ok, result, error, attempts, duration_ms}
  • Structured JSON logging of every agent start, success, and failure
  • In-memory agent trace for the current graph run (reset per invocation)
  • Metrics recording via observability.metrics
"""
from typing import Any, Callable, Optional, Tuple, Dict, List
import time
import traceback
import threading

# ── Observability imports (graceful degradation if not available) ─────────────
try:
    from app.observability.logger  import get_logger, get_trace_logger
    _log   = get_logger("medorchestrator.agent_runner")
    _trace = get_trace_logger()
except Exception:
    import logging
    _log   = logging.getLogger("medorchestrator.agent_runner")
    _trace = _log

try:
    from app.observability.metrics import collector as _metrics
except Exception:
    _metrics = None

# ── In-memory agent trace (reset per graph run) ───────────────────────────────
_agent_trace: List[Dict] = []
_trace_lock = threading.Lock()


def reset_agent_trace() -> None:
    global _agent_trace
    with _trace_lock:
        _agent_trace = []


def get_agent_trace() -> List[Dict]:
    with _trace_lock:
        return list(_agent_trace)


def _record_trace(entry: Dict) -> None:
    with _trace_lock:
        _agent_trace.append(entry)
    # Also write to the dedicated trace log file
    try:
        _trace.info(entry.get("agent", ""), extra=entry)
    except Exception:
        pass


# ── Core runner ───────────────────────────────────────────────────────────────

def call_agent(
    func:                Callable,
    args:                Tuple = (),
    kwargs:              Dict  = None,
    *,
    retries:             int   = 2,
    backoff:             float = 1.0,
    fallback:            Any   = None,
    swallow_exceptions:  bool  = True,
    node_name:           str   = "",
) -> Dict[str, Any]:
    """Call an agent function safely with retries, logging, and metrics.

    Parameters
    ----------
    func                : agent callable
    args / kwargs       : passed directly to func
    retries             : additional attempts after the first (total = retries+1)
    backoff             : base seconds for exponential backoff between retries
    fallback            : value (or callable) returned when all attempts fail
    swallow_exceptions  : if True, return error dict instead of re-raising
    node_name           : graph node label for trace / metrics

    Returns
    -------
    dict with keys:
      ok          – bool
      result      – return value of func (or fallback)
      error       – error message string (None when ok)
      attempts    – number of attempts performed
      duration_ms – total wall-clock time in milliseconds
    """
    if kwargs is None:
        kwargs = {}

    name       = node_name or getattr(func, "__name__", str(func))
    attempt    = 0
    last_exc   = None
    tb_str     = ""
    t_start    = time.perf_counter()

    # ── Record start ──────────────────────────────────────────────────────
    start_entry = {
        "event":   "agent_start",
        "agent":   name,
        "node":    node_name,
        "started_at": time.time(),
    }
    _record_trace(start_entry)
    _log.debug(f"Agent start: {name}", extra=start_entry)

    # ── Inform metrics about current agent ───────────────────────────────
    try:
        from app.config import set_calling_agent
        set_calling_agent(name)
    except Exception:
        pass

    # ── Attempt loop ──────────────────────────────────────────────────────
    while attempt <= retries:
        try:
            attempt += 1
            result = func(*args, **kwargs)

            duration_ms = (time.perf_counter() - t_start) * 1000
            success_entry = {
                "event":       "agent_success",
                "agent":       name,
                "node":        node_name,
                "attempts":    attempt,
                "duration_ms": round(duration_ms, 1),
                "ok":          True,
            }
            _record_trace(success_entry)
            _log.info(f"Agent OK: {name} ({round(duration_ms)}ms)", extra=success_entry)

            return {
                "ok":          True,
                "result":      result,
                "error":       None,
                "attempts":    attempt,
                "duration_ms": round(duration_ms, 1),
            }

        except Exception as exc:
            last_exc = exc
            tb_str   = traceback.format_exc()

            _log.warning(
                f"Agent attempt {attempt}/{retries+1} failed: {name} — {exc}",
                extra={
                    "event":   "agent_attempt_failed",
                    "agent":   name,
                    "node":    node_name,
                    "attempt": attempt,
                    "error":   str(exc),
                },
            )

            if _metrics:
                _metrics.record_agent_error(name, str(exc), node=node_name)

            if attempt > retries:
                break

            sleep_for = backoff * (2 ** (attempt - 1))
            time.sleep(sleep_for)

    # ── All attempts exhausted ────────────────────────────────────────────
    duration_ms = (time.perf_counter() - t_start) * 1000
    error_msg   = f"Agent '{name}' failed after {attempt} attempt(s): {last_exc}"

    failure_entry = {
        "event":       "agent_failed",
        "agent":       name,
        "node":        node_name,
        "attempts":    attempt,
        "duration_ms": round(duration_ms, 1),
        "ok":          False,
        "error":       str(last_exc),
    }
    _record_trace(failure_entry)
    _log.error(f"Agent FAILED: {name}", extra=failure_entry)

    # ── Compute fallback ──────────────────────────────────────────────────
    fallback_value = None
    try:
        fallback_value = fallback() if callable(fallback) else fallback
    except Exception as fe:
        _log.error(f"Fallback also failed for {name}: {fe}")

    if swallow_exceptions:
        return {
            "ok":          False,
            "result":      fallback_value,
            "error":       error_msg,
            "attempts":    attempt,
            "duration_ms": round(duration_ms, 1),
            "traceback":   tb_str,
        }
    raise last_exc
