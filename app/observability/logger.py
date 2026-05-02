"""Structured JSON logger for MedOrchestrator.

Usage:
    from app.observability.logger import get_logger
    logger = get_logger("medorchestrator.diagnosis")
    logger.info("Diagnosis completed", extra={"confidence": 0.82, "disease": "Influenza"})

All log records are written as newline-delimited JSON to logs/medorchestrator.jsonl.
Console only shows WARNING+ for clean terminal output; full DEBUG goes to file.
"""
import logging
import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# ── Log directory ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = _PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "medorchestrator.jsonl"
TRACE_FILE = LOG_DIR / "agent_trace.jsonl"


# ── JSON formatter ────────────────────────────────────────────────────────────

class _JSONFormatter(logging.Formatter):
    """Emit one JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        entry: Dict[str, Any] = {
            "ts":       datetime.now(timezone.utc).isoformat(),
            "level":    record.levelname,
            "logger":   record.name,
            "msg":      record.getMessage(),
            "module":   record.module,
            "fn":       record.funcName,
            "line":     record.lineno,
        }
        # Merge any `extra={}` kwargs passed by the caller
        for k, v in record.__dict__.items():
            if k not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            } and not k.startswith("_"):
                try:
                    json.dumps(v)           # only include JSON-serialisable values
                    entry[k] = v
                except (TypeError, ValueError):
                    entry[k] = str(v)

        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, ensure_ascii=False)


class _ColourFormatter(logging.Formatter):
    """Human-readable coloured formatter for the console."""
    _COLOURS = {
        "DEBUG":    "\033[36m",
        "INFO":     "\033[32m",
        "WARNING":  "\033[33m",
        "ERROR":    "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self._COLOURS.get(record.levelname, "")
        return (
            f"{colour}[{record.levelname[0]}]{self._RESET} "
            f"\033[90m{record.name}\033[0m — {record.getMessage()}"
        )


# ── Logger factory ────────────────────────────────────────────────────────────

_initialised: set = set()


def get_logger(name: str) -> logging.Logger:
    """Return (and lazily configure) a named logger."""
    logger = logging.getLogger(name)

    if name not in _initialised:
        _initialised.add(name)

        # ── File handler: DEBUG+ as JSON ────────────────────────────────
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a")
        fh.setFormatter(_JSONFormatter())
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

        # ── Console handler: WARNING+ in colour ─────────────────────────
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(_ColourFormatter())
        level_name = os.getenv("LOG_LEVEL", "WARNING").upper()
        ch.setLevel(getattr(logging, level_name, logging.WARNING))
        logger.addHandler(ch)

        logger.setLevel(logging.DEBUG)
        logger.propagate = False

    return logger


# ── Trace logger (separate file for agent transitions) ───────────────────────

def get_trace_logger() -> logging.Logger:
    """Dedicated logger for agent state-transition events."""
    name = "medorchestrator.trace"
    logger = logging.getLogger(name)
    if name not in _initialised:
        _initialised.add(name)
        fh = logging.FileHandler(TRACE_FILE, encoding="utf-8", mode="a")
        fh.setFormatter(_JSONFormatter())
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger


# ── Convenience helper ────────────────────────────────────────────────────────

def log_agent_transition(from_node: str, to_node: str, reason: str = "", **extra):
    """Log a graph node transition to the trace file."""
    get_trace_logger().info(
        f"{from_node} → {to_node}",
        extra={"from": from_node, "to": to_node, "reason": reason, **extra},
    )
