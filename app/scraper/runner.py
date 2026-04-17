"""Scraper Runner — orchestrates Scrapy spider execution from within the agent pipeline.

How it works:
1. Uses web_search to find the best starting URL for the hospital.
2. Runs the Scrapy spider as a subprocess (avoids Twisted reactor conflicts).
3. Reads the JSONL output and returns a list of clean doctor dicts.
4. Falls back gracefully if Scrapy / Playwright is not installed.
"""
import os
import sys
import json
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Directory containing scrapy.cfg (this file's directory)
_SCRAPER_DIR = Path(__file__).parent.resolve()

# Prefer the venv Python if available (ensures scrapy is on the path)
_VENV_PYTHON = _SCRAPER_DIR.parent.parent / "venv" / "bin" / "python"
_PYTHON_EXE  = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable


# ── URL discovery ─────────────────────────────────────────────────────────────

def _find_hospital_url(hospital_name: str, specialty: str, location: str) -> Optional[str]:
    """Search the web for the most relevant hospital doctors page URL."""
    try:
        # Avoid circular import — web_search lives in app.tools
        sys.path.insert(0, str(_SCRAPER_DIR.parent))
        from tools.web_search import web_search

        # Primary: look for doctors/specialists page directly
        queries = [
            f"{hospital_name} {location} {specialty} doctors specialists",
            f"{hospital_name} {location} official doctors page",
            f"{hospital_name} {location} site:*.com",
        ]
        for query in queries:
            results = web_search(query, max_results=5)
            for r in results:
                url = r.get("url", "")
                # Prefer URLs that are the hospital's own domain with doctor paths
                if url and any(
                    k in url.lower()
                    for k in ["doctor", "specialist", "physician", "team", "expert"]
                ):
                    return url
            # Fall back to first result
            if results:
                return results[0].get("url", "")
    except Exception as e:
        logger.warning(f"URL discovery failed: {e}")
    return None


# ── Subprocess runner ─────────────────────────────────────────────────────────

def scrape_doctors(
    hospital_name: str,
    specialty: str,
    location: str = "",
    start_url: Optional[str] = None,
    timeout: int = 90,
) -> List[Dict[str, Any]]:
    """Run the Scrapy doctor spider and return a list of DoctorItem dicts.

    Args:
        hospital_name : name of the hospital to scrape
        specialty     : target medical specialty (e.g. "cardiologist")
        location      : city/area to narrow search
        start_url     : override the auto-discovered URL
        timeout       : max seconds to wait for spider completion

    Returns:
        List of doctor dicts matching the fixed schema.
    """
    # ── Step 1: Find starting URL ──────────────────────────────────────────
    url = start_url or _find_hospital_url(hospital_name, specialty, location)
    if not url:
        logger.warning("Could not find a hospital URL to scrape.")
        return []

    logger.info(f"Starting doctor scrape: hospital={hospital_name!r} url={url!r}")

    # ── Step 2: Prepare temp output file ──────────────────────────────────
    with tempfile.NamedTemporaryFile(
        suffix=".jsonl", delete=False, mode="w", dir=_SCRAPER_DIR
    ) as tmp:
        output_path = tmp.name

    # ── Step 3: Build scrapy command ──────────────────────────────────────
    cmd = [
        _PYTHON_EXE, "-m", "scrapy", "crawl", "doctor_spider",
        "-a", f"start_url={url}",
        "-a", f"hospital_name={hospital_name}",
        "-a", f"specialty={specialty}",
        "-a", f"location={location}",
        "-o", output_path,
        "-t", "jsonlines",
        "-s", "LOG_LEVEL=ERROR",
        "-s", f"CLOSESPIDER_ITEMCOUNT=25",
        "-s", f"CLOSESPIDER_TIMEOUT=60",
    ]

    # ── Step 4: Run spider subprocess ─────────────────────────────────────
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_SCRAPER_DIR),
            capture_output=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode(errors="replace")
            logger.warning(f"Spider exited with code {proc.returncode}: {stderr[:500]}")
    except subprocess.TimeoutExpired:
        logger.warning("Spider timed out.")
    except FileNotFoundError:
        logger.error("Scrapy not found. Run: pip install scrapy scrapy-playwright && playwright install chromium")
        _cleanup(output_path)
        return []
    except Exception as e:
        logger.error(f"Spider subprocess error: {e}")
        _cleanup(output_path)
        return []

    # ── Step 5: Read JSONL results ─────────────────────────────────────────
    doctors: List[Dict[str, Any]] = []
    try:
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doctors.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    finally:
        _cleanup(output_path)

    logger.info(f"Scraped {len(doctors)} doctor(s) from {url}")
    return doctors


def _cleanup(path: str):
    try:
        os.unlink(path)
    except Exception:
        pass
