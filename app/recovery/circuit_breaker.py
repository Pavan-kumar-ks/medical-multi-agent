"""Circuit breaker pattern for external service calls.

Prevents cascade failures when third-party APIs (Groq, web search, maps)
are temporarily unavailable.

State machine:
  CLOSED    — normal operation, calls pass through
  OPEN      — failure threshold exceeded; calls are immediately rejected
              with the fallback value / exception
  HALF_OPEN — after recovery_timeout, one probe call is allowed:
              success → CLOSED; failure → OPEN again

Usage:
    from app.recovery.circuit_breaker import web_search_breaker

    result = web_search_breaker.call(
        web_search, "nearest hospital", max_results=5,
        fallback=[]
    )
"""
import time
import threading
import logging
from enum import Enum
from typing import Any, Callable, Optional

try:
    from app.observability.logger import get_logger
    _log = get_logger("medorchestrator.circuit_breaker")
except Exception:
    _log = logging.getLogger("medorchestrator.circuit_breaker")


# ── State ─────────────────────────────────────────────────────────────────────

class _State(Enum):
    CLOSED    = "CLOSED"
    OPEN      = "OPEN"
    HALF_OPEN = "HALF_OPEN"


# ── Circuit breaker ───────────────────────────────────────────────────────────

class CircuitBreaker:
    """Thread-safe circuit breaker.

    Parameters
    ----------
    name              : human-readable name for logging
    failure_threshold : consecutive failures before opening the circuit
    recovery_timeout  : seconds to wait before entering HALF_OPEN state
    """

    def __init__(
        self,
        name:              str,
        failure_threshold: int   = 5,
        recovery_timeout:  float = 60.0,
    ):
        self.name              = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout  = recovery_timeout

        self._state:         _State = _State.CLOSED
        self._failure_count: int    = 0
        self._last_failure:  float  = 0.0
        self._lock                  = threading.Lock()

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state.value

    @property
    def is_open(self) -> bool:
        return self._state == _State.OPEN

    # ── Core call ─────────────────────────────────────────────────────────

    def call(
        self,
        func:     Callable,
        *args,
        fallback: Any = None,
        **kwargs,
    ) -> Any:
        """Execute `func(*args, **kwargs)` through the circuit breaker.

        If the circuit is OPEN and the recovery timeout has not elapsed,
        `fallback` is returned immediately (or called if callable).
        """
        with self._lock:
            current_state = self._state

            if current_state == _State.OPEN:
                elapsed = time.time() - self._last_failure
                if elapsed >= self.recovery_timeout:
                    self._state = _State.HALF_OPEN
                    _log.info(f"CircuitBreaker [{self.name}] → HALF_OPEN (probe)")
                else:
                    _log.warning(
                        f"CircuitBreaker [{self.name}] OPEN — fast-fail "
                        f"({int(self.recovery_timeout - elapsed)}s to probe)",
                        extra={"breaker": self.name, "state": "OPEN"},
                    )
                    return fallback() if callable(fallback) else fallback

        # Try the call (CLOSED or HALF_OPEN)
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as exc:
            self._on_failure(exc)
            raise

    # ── State transitions ─────────────────────────────────────────────────

    def _on_success(self) -> None:
        with self._lock:
            if self._state == _State.HALF_OPEN:
                _log.info(f"CircuitBreaker [{self.name}] → CLOSED (probe succeeded)")
            self._state         = _State.CLOSED
            self._failure_count = 0

    def _on_failure(self, exc: Exception) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure   = time.time()

            if self._state == _State.HALF_OPEN:
                self._state = _State.OPEN
                _log.error(
                    f"CircuitBreaker [{self.name}] probe FAILED → OPEN again",
                    extra={"breaker": self.name, "error": str(exc)},
                )
            elif self._failure_count >= self.failure_threshold:
                self._state = _State.OPEN
                _log.error(
                    f"CircuitBreaker [{self.name}] threshold reached → OPEN "
                    f"({self._failure_count} failures)",
                    extra={"breaker": self.name, "failures": self._failure_count},
                )
            else:
                _log.warning(
                    f"CircuitBreaker [{self.name}] failure "
                    f"{self._failure_count}/{self.failure_threshold}: {exc}",
                    extra={"breaker": self.name},
                )

    def reset(self) -> None:
        with self._lock:
            self._state         = _State.CLOSED
            self._failure_count = 0
            self._last_failure  = 0.0
        _log.info(f"CircuitBreaker [{self.name}] manually RESET")

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name={self.name!r}, "
            f"state={self._state.value}, "
            f"failures={self._failure_count})"
        )


# ── Pre-built breakers for each external dependency ───────────────────────────

llm_breaker        = CircuitBreaker("groq_llm",    failure_threshold=3, recovery_timeout=30)
web_search_breaker = CircuitBreaker("web_search",  failure_threshold=5, recovery_timeout=60)
maps_breaker       = CircuitBreaker("maps_api",    failure_threshold=4, recovery_timeout=120)
scraper_breaker    = CircuitBreaker("web_scraper", failure_threshold=3, recovery_timeout=45)
