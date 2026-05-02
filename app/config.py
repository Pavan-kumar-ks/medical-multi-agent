"""LLM client configuration for MedOrchestrator.

llm_call() wraps Groq with:
  • Exponential-backoff retry with jitter (3 attempts)
  • Token usage + latency metrics via observability.metrics
  • Structured logging via observability.logger
  • Hallucination detection (logged, not raised)
  • Context overflow protection (truncates oversized prompts)
"""
import os
import time
import random
import logging
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama3-8b-8192")

# ── Groq client ───────────────────────────────────────────────────────────────
client = Groq(api_key=GROQ_API_KEY)

# ── Logger ────────────────────────────────────────────────────────────────────
try:
    from app.observability.logger import get_logger
    _log = get_logger("medorchestrator.llm")
except Exception:
    _log = logging.getLogger("medorchestrator.llm")

# ── Constants ─────────────────────────────────────────────────────────────────
_MAX_PROMPT_CHARS  = 12_000   # ~3 000 tokens; truncate beyond this
_MAX_RETRIES       = 3
_BACKOFF_BASE      = 1.5      # seconds
_BACKOFF_JITTER    = 0.4      # ± random seconds

# Context variable: which agent is currently calling (set by each agent node)
_current_agent: str = "unknown"


def set_calling_agent(name: str) -> None:
    """Tell the metrics collector which agent owns the next llm_call."""
    global _current_agent
    _current_agent = name


def _truncate_prompt(prompt: str) -> str:
    """Truncate oversized prompts to avoid context-window overflow errors."""
    if len(prompt) <= _MAX_PROMPT_CHARS:
        return prompt
    _log.warning(
        "Prompt truncated",
        extra={"original_len": len(prompt), "truncated_to": _MAX_PROMPT_CHARS, "agent": _current_agent},
    )
    # Keep the start (instructions) and end (recent context) — drop the middle
    half = _MAX_PROMPT_CHARS // 2
    return (
        prompt[:half]
        + "\n\n[... content truncated for context limit ...]\n\n"
        + prompt[-half:]
    )


def llm_call(prompt: str, agent: Optional[str] = None) -> str:
    """Call the Groq LLM with full observability and retry logic.

    Parameters
    ----------
    prompt : str
        The full prompt string.
    agent : str, optional
        Override the calling-agent name for metrics labelling.

    Returns
    -------
    str
        The model's text response.

    Raises
    ------
    RuntimeError
        If all retry attempts fail.
    """
    # Import here to avoid circular import at module load time
    try:
        from app.observability.metrics import collector
    except Exception:
        collector = None

    try:
        from app.recovery.hallucination_guard import detect_hallucination
        _guard_available = True
    except Exception:
        _guard_available = False

    calling_agent = agent or _current_agent
    prompt = _truncate_prompt(prompt)

    last_error: Optional[Exception] = None

    for attempt in range(1, _MAX_RETRIES + 1):
        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            content    = response.choices[0].message.content
            latency_ms = (time.perf_counter() - t0) * 1000

            # Token usage
            usage      = response.usage
            tokens_in  = getattr(usage, "prompt_tokens",     0) if usage else 0
            tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0

            # ── Metrics ────────────────────────────────────────────────
            if collector:
                collector.record_llm_call(
                    agent      = calling_agent,
                    model      = MODEL_NAME,
                    tokens_in  = tokens_in,
                    tokens_out = tokens_out,
                    latency_ms = latency_ms,
                    success    = True,
                )

            _log.debug(
                "LLM call OK",
                extra={
                    "agent":      calling_agent,
                    "attempt":    attempt,
                    "tokens_in":  tokens_in,
                    "tokens_out": tokens_out,
                    "latency_ms": round(latency_ms, 1),
                    "model":      MODEL_NAME,
                },
            )

            # ── Hallucination check (non-blocking) ─────────────────────
            if _guard_available:
                try:
                    flagged, flags = detect_hallucination(content, prompt)
                    if flagged:
                        _log.warning(
                            "Hallucination flags detected",
                            extra={"agent": calling_agent, "flags": flags},
                        )
                except Exception:
                    pass

            return content

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            last_error = exc

            if collector:
                collector.record_llm_call(
                    agent      = calling_agent,
                    model      = MODEL_NAME,
                    tokens_in  = 0,
                    tokens_out = 0,
                    latency_ms = latency_ms,
                    success    = False,
                    error      = str(exc),
                )

            _log.warning(
                f"LLM call failed (attempt {attempt}/{_MAX_RETRIES})",
                extra={"agent": calling_agent, "error": str(exc)},
            )

            if attempt < _MAX_RETRIES:
                sleep = (_BACKOFF_BASE ** attempt) + random.uniform(0, _BACKOFF_JITTER)
                time.sleep(sleep)

    raise RuntimeError(
        f"llm_call failed after {_MAX_RETRIES} attempts for agent '{calling_agent}': {last_error}"
    )
