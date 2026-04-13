"""Safe agent invocation helper.

Provides `call_agent` which wraps agent function calls with:
- try/except capturing exceptions
- configurable retries with exponential backoff
- optional fallback value/function returned on failure
- standardized return shape indicating success/error and details

Usage example:

from app.orchestrator.agent_runner import call_agent

result = call_agent(my_agent_function, args=(patient,), kwargs={}, retries=2, fallback={'status':'failed'})

The returned dict has keys: `ok` (bool), `result` (agent return or None), `error` (error message or None), `attempts` (int)
"""
from typing import Any, Callable, Optional, Tuple, Dict
import time
import traceback
import time as _time

# Simple in-memory trace of agent executions for observability during a single run
_agent_trace = []


def reset_agent_trace():
    global _agent_trace
    _agent_trace = []


def get_agent_trace():
    return list(_agent_trace)


def call_agent(func: Callable, args: Tuple = (), kwargs: Dict = None, *, retries: int = 2, backoff: float = 1.0, fallback: Any = None, swallow_exceptions: bool = True) -> Dict[str, Any]:
    """Call an agent function safely.

    Parameters
    - func: callable to invoke
    - args, kwargs: passed to func
    - retries: number of retries after the initial attempt (total attempts = retries + 1)
    - backoff: multiplier for exponential backoff (seconds)
    - fallback: value or callable to return if all attempts fail
    - swallow_exceptions: if True, do not re-raise exceptions (returns structured error)

    Returns a dict with:
    - `ok`: bool
    - `result`: the function return value (when ok)
    - `error`: error string when not ok
    - `attempts`: number of attempts performed
    - `trace`: optional traceback string for debugging
    """
    if kwargs is None:
        kwargs = {}

    attempt = 0
    last_exc = None
    start_time = _time.time()
    # record agent start
    try:
        name = getattr(func, "__name__", str(func))
    except Exception:
        name = str(func)
    _agent_trace.append({"agent": name, "started_at": start_time, "attempts": 0, "ok": None})

    while attempt <= retries:
        try:
            attempt += 1
            res = func(*args, **kwargs)
            # update trace record
            try:
                _agent_trace[-1]["attempts"] = attempt
                _agent_trace[-1]["ok"] = True
                _agent_trace[-1]["ended_at"] = _time.time()
            except Exception:
                pass
            return {"ok": True, "result": res, "error": None, "attempts": attempt}
        except Exception as e:
            last_exc = e
            tb = traceback.format_exc()
            # Simple logging to stderr — in real app replace with structured logger
            print(f"[agent_runner] attempt={attempt} func={getattr(func, '__name__', str(func))} error={e}")
            print(tb)
            if attempt > retries:
                break
            # Sleep with exponential backoff
            sleep_for = backoff * (2 ** (attempt - 1))
            time.sleep(sleep_for)

    # All attempts failed
    error_msg = f"Agent {getattr(func, '__name__', str(func))} failed after {attempt} attempts: {last_exc}"

    # Compute fallback result
    fallback_value = None
    try:
        if callable(fallback):
            fallback_value = fallback()
        else:
            fallback_value = fallback
    except Exception as fe:
        # If fallback also fails, include that info in trace but continue
        fallback_value = None
        print(f"[agent_runner] fallback failed: {fe}")

    if swallow_exceptions:
        try:
            _agent_trace[-1]["attempts"] = attempt
            _agent_trace[-1]["ok"] = False
            _agent_trace[-1]["ended_at"] = _time.time()
            _agent_trace[-1]["error"] = str(last_exc)
        except Exception:
            pass
        return {"ok": False, "result": fallback_value, "error": error_msg, "attempts": attempt, "trace": tb}
    else:
        raise last_exc
