import json
from typing import List, Dict, Any

from app.config import llm_call


def _safe_parse_json(response: str):
    try:
        return json.loads(response)
    except Exception:
        # best-effort extract
        s = response.find("[")
        e = response.rfind("]")
        if s != -1 and e != -1:
            try:
                return json.loads(response[s:e+1])
            except Exception:
                pass
    return None


def fetch_emergency_contacts(state: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Ask the LLM for a short list of local emergency contact numbers.

    Returns a list of objects: [{"label": "Ambulance", "number": "100"}, ...]
    """
    # Build a small prompt that instructs the model to return JSON
    location_hint = None
    try:
        patient = state.get("patient", {})
        # attempt to extract country or location if present (future)
        location_hint = patient.get("location")
    except Exception:
        location_hint = None

    prompt = "Provide 3 short emergency contact numbers (label and number) in JSON array format."
    if location_hint:
        prompt += f" Prefer numbers relevant to this location: {location_hint}."
    prompt += " Example output:\n[{\"label\": \"Ambulance\", \"number\": \"100\"}]\nReturn ONLY valid JSON."

    response = llm_call(prompt)
    parsed = _safe_parse_json(response)
    if isinstance(parsed, list):
        out = []
        for item in parsed:
            try:
                lbl = item.get("label") if isinstance(item, dict) else None
                num = item.get("number") if isinstance(item, dict) else None
                if lbl and num:
                    out.append({"label": str(lbl), "number": str(num)})
            except Exception:
                continue
        return out
    return []
