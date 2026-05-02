"""Map provider utilities (Mappls with OSM fallback).

Default behavior remains OSM-based. Set MAP_PROVIDER=mappls to enable
Mappls APIs for geocoding and nearby hospital search.
"""
from typing import Dict, List, Any, Optional
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

MAP_PROVIDER = os.getenv("MAP_PROVIDER", "osm").strip().lower()
MAPPLS_SECRET_KEY = os.getenv("MAPPLS_SECRET_KEY", "").strip() or os.getenv("MAPPLS_CLIENT_SECRET", "").strip()
MAPPLS_GEOCODE_URL = os.getenv("MAPPLS_GEOCODE_URL", "https://atlas.mappls.com/api/places/geocode")
MAPPLS_NEARBY_URL = os.getenv("MAPPLS_NEARBY_URL", "https://atlas.mappls.com/api/places/nearby/json")


def _headers() -> Dict[str, str]:
    # Nominatim requires a valid user-agent; provide contact info via env if available
    contact = os.getenv("NOMINATIM_EMAIL", "medical-multi-agent")
    return {"User-Agent": f"medical-multi-agent/1.0 ({contact})"}


def _mappls_enabled() -> bool:
    return MAP_PROVIDER == "mappls" and bool(MAPPLS_SECRET_KEY)


def _mappls_auth_headers() -> Dict[str, str]:
    if not MAPPLS_SECRET_KEY:
        return {}
    # Different Mappls plans/products can expect different auth styles.
    # Provide the common forms using the same secret key.
    return {
        "Authorization": f"Bearer {MAPPLS_SECRET_KEY}",
        "x-api-key": MAPPLS_SECRET_KEY,
    }


def _parse_mappls_lat_lng(item: Dict[str, Any]) -> Optional[Dict[str, float]]:
    lat_keys = ("latitude", "lat", "y")
    lng_keys = ("longitude", "lng", "lon", "x")

    lat = None
    lng = None
    for k in lat_keys:
        if item.get(k) is not None:
            lat = item.get(k)
            break
    for k in lng_keys:
        if item.get(k) is not None:
            lng = item.get(k)
            break

    if lat is None or lng is None:
        return None

    try:
        return {"lat": float(lat), "lng": float(lng)}
    except Exception:
        return None


def geocode_location(location_text: str) -> Dict[str, Any]:
    """Return a dict with lat/lng and formatted address.

    Uses Mappls geocode when enabled; falls back to Nominatim.
    """
    if not location_text:
        return {}

    if _mappls_enabled():
        try:
            resp = requests.get(
                MAPPLS_GEOCODE_URL,
                params={
                    "address": location_text,
                    "key": MAPPLS_SECRET_KEY,
                },
                headers=_mappls_auth_headers(),
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json() or {}
            candidates = []
            if isinstance(data, list):
                candidates = data
            elif isinstance(data, dict):
                candidates = data.get("results") or data.get("copResults") or data.get("suggestedLocations") or []
            if candidates:
                top = candidates[0] or {}
                coords = _parse_mappls_lat_lng(top)
                if coords:
                    formatted = (
                        top.get("formatted_address")
                        or top.get("placeName")
                        or top.get("placeAddress")
                        or top.get("address")
                        or location_text
                    )
                    return {
                        "lat": coords["lat"],
                        "lng": coords["lng"],
                        "formatted": formatted,
                        "mappls_id": top.get("eLoc") or top.get("mapplsPin"),
                    }
        except Exception:
            pass

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
    """Return a list of nearby hospitals.

    Uses Mappls nearby search when enabled; falls back to Overpass.
    """
    if _mappls_enabled():
        try:
            resp = requests.get(
                MAPPLS_NEARBY_URL,
                params={
                    "key": MAPPLS_SECRET_KEY,
                    "keywords": "hospital",
                    "refLocation": f"{lat},{lng}",
                    "radius": radius_m,
                },
                headers=_mappls_auth_headers(),
                timeout=25,
            )
            resp.raise_for_status()
            data = resp.json() or {}
            raw_items = []
            if isinstance(data, list):
                raw_items = data
            elif isinstance(data, dict):
                raw_items = data.get("suggestedLocations") or data.get("results") or data.get("items") or []

            out: List[Dict[str, Any]] = []
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                coords = _parse_mappls_lat_lng(item) or {}
                out.append({
                    "name": item.get("placeName") or item.get("poi") or item.get("name") or "Hospital",
                    "address": item.get("placeAddress") or item.get("address") or "",
                    "phone": item.get("telNo") or item.get("phone") or "",
                    "lat": coords.get("lat"),
                    "lng": coords.get("lng"),
                    "place_id": item.get("eLoc") or item.get("mapplsPin") or item.get("id") or "",
                    "distance_m": item.get("distance") if isinstance(item.get("distance"), (int, float)) else None,
                })
            if out:
                return out
        except Exception:
            pass

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


def reverse_geocode(lat: float, lng: float) -> Dict[str, Any]:
    """Reverse geocode lat/lng → human-readable address via Nominatim.

    Returns a dict with keys: ``short`` (neighbourhood/town), ``full`` (display_name).
    Falls back to empty strings on failure.
    """
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lng, "format": "json"},
            headers=_headers(),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        address = data.get("address") or {}

        # Build a short human-readable label from the most specific fields
        parts = []
        for field in ("suburb", "neighbourhood", "village", "town", "city_district",
                      "county", "city", "state_district", "state"):
            val = address.get(field)
            if val and val not in parts:
                parts.append(val)
            if len(parts) == 3:
                break

        short = ", ".join(parts) if parts else data.get("display_name", "")
        return {
            "short":   short,
            "full":    data.get("display_name", ""),
            "address": address,
        }
    except Exception:
        return {"short": "", "full": "", "address": {}}


def get_place_details(place_id: str) -> Dict[str, Any]:
    """Return details if available (OSM data already includes tags)."""
    return {}


def get_travel_time(origin_lat: float, origin_lng: float, dest_lat: float, dest_lng: float) -> Dict[str, Any]:
    """Return travel time and distance.

    For now this uses OSRM even when Mappls is enabled to keep routing stable
    without requiring additional paid endpoint configuration.
    """
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
