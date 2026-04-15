"""OpenStreetMap stack (Nominatim + Overpass + OSRM).

This module implements geocoding, nearby hospital search, and travel time
without any paid API requirements.
"""
from typing import Dict, List, Any
import os
import requests

NOMINATIM_URL = os.getenv("NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")
OVERPASS_URL = os.getenv("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
OVERPASS_FALLBACK_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
]
OSRM_URL = os.getenv("OSRM_URL", "http://router.project-osrm.org/route/v1/driving")


def _headers() -> Dict[str, str]:
    # Nominatim requires a valid user-agent; provide contact info via env if available
    contact = os.getenv("NOMINATIM_EMAIL", "medical-multi-agent")
    return {"User-Agent": f"medical-multi-agent/1.0 ({contact})"}


def geocode_location(location_text: str) -> Dict[str, Any]:
    """Return a dict with lat/lng and formatted address via Nominatim."""
    if not location_text:
        return {}
    params = {
        "q": location_text,
        "format": "json",
        "limit": 1,
    }
    resp = requests.get(NOMINATIM_URL, params=params, headers=_headers(), timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return {}
    top = data[0]
    return {
        "lat": float(top.get("lat")),
        "lng": float(top.get("lon")),
        "formatted": top.get("display_name"),
        "osm_id": top.get("osm_id"),
        "osm_type": top.get("osm_type"),
    }


def find_nearby_hospitals(lat: float, lng: float, radius_m: int = 5000) -> List[Dict[str, Any]]:
    """Return a list of nearby hospitals using Overpass API."""
    # Overpass query (around radius in meters)
    query = f"""
[out:json];
(
  node["amenity"="hospital"](around:{radius_m},{lat},{lng});
  way["amenity"="hospital"](around:{radius_m},{lat},{lng});
  relation["amenity"="hospital"](around:{radius_m},{lat},{lng});
);
out center;
"""
    data = None
    urls = [OVERPASS_URL] + [u for u in OVERPASS_FALLBACK_URLS if u != OVERPASS_URL]
    last_err = None
    for url in urls:
        try:
            resp = requests.get(url, params={"data": query}, headers=_headers(), timeout=30)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            last_err = e
            continue
    if data is None:
        # All endpoints failed; return empty list to keep flow running
        return []
    results: List[Dict[str, Any]] = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        # center for ways/relations, lat/lon for nodes
        if "lat" in el and "lon" in el:
            el_lat, el_lng = el.get("lat"), el.get("lon")
        else:
            center = el.get("center") or {}
            el_lat, el_lng = center.get("lat"), center.get("lon")

        results.append({
            "name": tags.get("name") or "Hospital",
            "address": tags.get("addr:full") or tags.get("addr:street") or "",
            "phone": tags.get("phone") or tags.get("contact:phone") or "",
            "lat": el_lat,
            "lng": el_lng,
            "place_id": f"osm:{el.get('type')}:{el.get('id')}",
        })
    return results


def get_place_details(place_id: str) -> Dict[str, Any]:
    """Return details if available (OSM data already includes tags)."""
    return {}


def get_travel_time(origin_lat: float, origin_lng: float, dest_lat: float, dest_lng: float) -> Dict[str, Any]:
    """Return travel time and distance using OSRM."""
    url = f"{OSRM_URL}/{origin_lng},{origin_lat};{dest_lng},{dest_lat}"
    params = {"overview": "false"}
    resp = requests.get(url, params=params, headers=_headers(), timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("routes"):
        return {}
    route = data["routes"][0]
    return {
        "distance_m": route.get("distance"),
        "duration_s": route.get("duration"),
    }
