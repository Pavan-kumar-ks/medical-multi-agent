from typing import Dict, Any, List

from app.tools.mcp_maps import geocode_location, find_nearby_hospitals, get_place_details, get_travel_time


def hospital_finder_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Find nearby hospitals within 5km of the user's location.

    Expects: state["location"] = {"text": "..."} or {"lat": .., "lng": ..}
    Returns: {"hospitals": [ ... ]}
    """
    loc = state.get("location") or {}
    lat = loc.get("lat")
    lng = loc.get("lng")

    if lat is None or lng is None:
        # Try geocoding using MCP wrapper
        text = loc.get("text") or ""
        if text:
            geo = geocode_location(text)
            lat = geo.get("lat")
            lng = geo.get("lng")
            if lat is not None and lng is not None:
                loc.update({"lat": lat, "lng": lng, "formatted": geo.get("formatted")})

    if lat is None or lng is None:
        return {"hospitals": []}

    try:
        results = find_nearby_hospitals(lat, lng, radius_m=5000)
    except Exception:
        results = []
    hospitals: List[Dict[str, Any]] = []
    for r in results[:5]:
        place_id = r.get("place_id")
        details = get_place_details(place_id) if place_id else {}
        # compute travel time if coordinates are available
        travel = {}
        try:
            if r.get("lat") is not None and r.get("lng") is not None:
                travel = get_travel_time(lat, lng, r.get("lat"), r.get("lng"))
        except Exception:
            travel = {}
        hospitals.append({
            "name": r.get("name"),
            "address": details.get("address") or r.get("address"),
            "phone": details.get("phone"),
            "distance_m": travel.get("distance_m") or r.get("distance_m"),
            "travel_time_s": travel.get("duration_s"),
            "place_id": place_id,
        })

    return {"hospitals": hospitals}
