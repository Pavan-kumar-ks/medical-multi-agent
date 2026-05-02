"""Web search utility for the medical multi-agent system.

Uses duckduckgo_search library (no API key required).
Falls back to DuckDuckGo instant-answer API if library unavailable.
"""
import re
import json
from typing import List, Dict, Any

import requests

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _ddgs_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Try both the new `ddgs` package and the legacy `duckduckgo_search` package."""
    # ── Try new package name: ddgs ────────────────────────────────────────
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                from ddgs import DDGS  # type: ignore  # new name
            except ImportError:
                from duckduckgo_search import DDGS  # type: ignore  # legacy name
        with DDGS() as client:
            raw = list(client.text(query, max_results=max_results))
        return [
            {
                "title":   r.get("title", ""),
                "url":     r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]
    except ImportError:
        pass
    except Exception:
        pass
    return []


def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search the web and return [{title, url, snippet}].

    Tries (in order):
    1. ddgs / duckduckgo_search library
    2. DuckDuckGo instant-answer JSON API
    """
    # ── Attempt 1: DDGS library ───────────────────────────────────────────
    results = _ddgs_search(query, max_results)
    if results:
        return results

    # ── Attempt 2: DuckDuckGo instant-answer API ──────────────────────────
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            headers=_HEADERS,
            timeout=10,
        )
        data = resp.json()
        results: List[Dict[str, Any]] = []

        if data.get("Abstract"):
            results.append({
                "title":   data.get("Heading", query),
                "url":     data.get("AbstractURL", ""),
                "snippet": data.get("Abstract", ""),
            })

        for topic in data.get("RelatedTopics", []):
            if len(results) >= max_results:
                break
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title":   topic.get("Text", "")[:100],
                    "url":     topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", ""),
                })

        return results[:max_results]
    except Exception:
        pass

    return []


def fetch_page_snippet(url: str, max_chars: int = 2000) -> str:
    """Fetch a URL and return stripped plain text (best-effort)."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        text = re.sub(r"<[^>]+>", " ", resp.text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception:
        return ""
