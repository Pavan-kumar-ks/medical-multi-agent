import json
from typing import Dict, Any, List

from app.tools.mcp_maps import geocode_location
from app.config import llm_call


def location_intake_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract or request location before diagnosis.

    Returns:
    - location: {"text": str}
    - need_location: bool
    - location_prompt: str
    """
    session = state.get("session_memory") or {}

    # If we already have location stored, return it
    existing = session.get("location")
    if isinstance(existing, dict) and existing.get("text"):
        return {"location": existing, "need_location": False}

    user_input = state.get("user_input", "").strip()

    # If we were awaiting location, try to resolve it via OSM geocode
    if session.get("awaiting_location"):
        location_text = user_input
        variants: List[str] = [location_text]

        # Semantic fallback: ask LLM for alternative location variants
        try:
            prompt = f"""
You are normalizing a user location for geocoding. Return ONLY JSON with a key "variants":
{{"variants": ["normalized location", "city, neighborhood", "landmark, city"]}}
User input: {location_text}
"""
            response = llm_call(prompt)
            data = json.loads(response)
            llm_vars = data.get("variants") if isinstance(data, dict) else None
            if isinstance(llm_vars, list):
                variants.extend([v for v in llm_vars if isinstance(v, str)])
        except Exception:
            pass

        # Try each variant until a geocode works
        for v in variants:
            geo = {}
            try:
                geo = geocode_location(v)
            except Exception:
                geo = {}

            if geo and geo.get("lat") is not None and geo.get("lng") is not None:
                session["location_candidate"] = {
                    "text": v,
                    "lat": geo.get("lat"),
                    "lng": geo.get("lng"),
                    "formatted": geo.get("formatted"),
                }
                session["awaiting_location"] = False
                session["confirm_location"] = True
                return {
                    "location_candidate": session["location_candidate"],
                    "need_location": False,
                    "confirm_location": True,
                    "session_memory": session,
                }

    # Otherwise, request location from the user
    session["awaiting_location"] = True
    return {
        "need_location": True,
        "location_prompt": "Please share your current location (city, neighborhood, or a nearby landmark).",
        "session_memory": session,
    }
