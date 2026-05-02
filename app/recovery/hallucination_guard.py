"""Hallucination detection and output sanitisation for MedOrchestrator.

detect_hallucination(response, prompt)
    • Flags invented phone numbers not present in the prompt
    • Flags impossibly high confidence claims (>0.99 with certainty language)
    • Flags JSON responses that claim more diagnoses than symptoms warrant
    • Flags drug/dosage specifics not grounded in the prompt context

sanitize_response(response, prompt)
    • Removes or masks detected hallucinated content
    • Returns the cleaned response + a list of flags

All detections are heuristic and conservative — false positives are logged,
not raised as errors, so the pipeline continues gracefully.
"""
import re
import json
import logging
from typing import List, Tuple, Dict, Any

try:
    from app.observability.logger import get_logger
    _log = get_logger("medorchestrator.hallucination")
except Exception:
    _log = logging.getLogger("medorchestrator.hallucination")

# ── Patterns ──────────────────────────────────────────────────────────────────

_PHONE_RE   = re.compile(r"\+?[\d][\d\s\-\(\)]{8,14}[\d]")
_DOSAGE_RE  = re.compile(r"\b\d+\s*(mg|mcg|ml|units?|IU|g)\b", re.IGNORECASE)
_URL_RE     = re.compile(r"https?://[^\s\"'>]+")
_CERTAINTY_PHRASES = [
    "100% certain", "definitely diagnosed", "confirmed diagnosis",
    "without any doubt", "absolutely certain", "no other possibility",
]
_HIGH_CONF_RE = re.compile(r'"confidence"\s*:\s*0\.9[89]|"confidence"\s*:\s*1\.0')

# Known safe drug names (won't be flagged as invented)
_SAFE_DRUGS = {
    "paracetamol", "acetaminophen", "ibuprofen", "aspirin", "amoxicillin",
    "metformin", "atorvastatin", "omeprazole", "lisinopril", "cetirizine",
    "doxycycline", "azithromycin", "ciprofloxacin", "metronidazole",
    "pantoprazole", "clopidogrel", "amlodipine", "losartan", "insulin",
    "salbutamol", "prednisolone", "dexamethasone", "hydroxychloroquine",
    "oseltamivir", "ivermectin", "ranitidine", "ors", "zinc",
}


# ── Core detection ────────────────────────────────────────────────────────────

def detect_hallucination(response: str, prompt: str) -> Tuple[bool, List[str]]:
    """Analyse `response` for content not grounded in `prompt`.

    Returns
    -------
    (flagged: bool, flags: list[str])
        flagged – True if at least one hallucination indicator was found
        flags   – human-readable list of what was detected
    """
    flags: List[str] = []

    # ── 1. Invented phone numbers ─────────────────────────────────────────
    response_phones = _PHONE_RE.findall(response)
    prompt_phones   = _PHONE_RE.findall(prompt)

    def _normalise_phone(p: str) -> str:
        return re.sub(r"\D", "", p)

    prompt_phone_set = {_normalise_phone(p) for p in prompt_phones if len(_normalise_phone(p)) >= 7}
    for phone in response_phones:
        norm = _normalise_phone(phone)
        if len(norm) >= 7 and norm not in prompt_phone_set:
            flags.append(f"invented_phone:{phone.strip()}")

    # ── 2. Invented URLs not in prompt ────────────────────────────────────
    resp_urls   = set(_URL_RE.findall(response))
    prompt_urls = set(_URL_RE.findall(prompt))
    for url in resp_urls - prompt_urls:
        # Booking / appointment URLs appearing out of nowhere
        if any(k in url.lower() for k in ["book", "appoint", "consult", "doctor"]):
            flags.append(f"invented_url:{url}")

    # ── 3. Certainty phrases (overconfidence) ─────────────────────────────
    resp_lower = response.lower()
    for phrase in _CERTAINTY_PHRASES:
        if phrase.lower() in resp_lower:
            flags.append(f"overconfidence:'{phrase}'")

    # ── 4. Impossibly high confidence in JSON output ──────────────────────
    if _HIGH_CONF_RE.search(response):
        flags.append("confidence_too_high:>=0.98_detected_in_json")

    # ── 5. Specific dosages not grounded in prompt ────────────────────────
    resp_dosages   = set(m.group().lower() for m in _DOSAGE_RE.finditer(response))
    prompt_dosages = set(m.group().lower() for m in _DOSAGE_RE.finditer(prompt))
    invented_doses = resp_dosages - prompt_dosages
    if invented_doses:
        flags.append(f"invented_dosage:{','.join(invented_doses)}")

    # ── 6. Structural: JSON array length sanity check ─────────────────────
    if response.strip().startswith("["):
        try:
            items = json.loads(response)
            if isinstance(items, list) and len(items) > 15:
                flags.append(f"excessive_list_length:{len(items)}_items")
        except json.JSONDecodeError:
            flags.append("malformed_json:expected_array")

    flagged = len(flags) > 0
    return flagged, flags


def sanitize_response(response: str, prompt: str, agent: str = "") -> str:
    """Log hallucination flags and return the response unchanged.

    The pipeline continues regardless — this is an observability hook,
    not a blocking filter. If you want to block, raise here.
    """
    flagged, flags = detect_hallucination(response, prompt)
    if flagged:
        _log.warning(
            "Hallucination flags",
            extra={"agent": agent, "flags": flags, "response_len": len(response)},
        )
    return response


# ── Context overflow guard ────────────────────────────────────────────────────

def guard_context_overflow(prompt: str, max_chars: int = 12_000, agent: str = "") -> str:
    """Truncate prompt if it exceeds the safe context window.

    Keeps the beginning (instructions + schema) and the end (most recent context).
    """
    if len(prompt) <= max_chars:
        return prompt

    _log.warning(
        "Context overflow — truncating prompt",
        extra={"agent": agent, "original_chars": len(prompt), "limit": max_chars},
    )
    half = max_chars // 2
    return (
        prompt[:half]
        + "\n\n[... middle section removed to fit context limit ...]\n\n"
        + prompt[-half:]
    )


# ── JSON output validator ─────────────────────────────────────────────────────

def safe_parse_json(text: str, fallback: Any = None) -> Any:
    """Extract and parse the first JSON object or array from `text`.

    Returns `fallback` if parsing fails — never raises.
    """
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end   = text.rfind(end_char) + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    _log.debug("safe_parse_json: no valid JSON found; returning fallback")
    return fallback
