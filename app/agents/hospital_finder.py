from typing import Dict, Any, List
import math

from app.tools.mcp_maps import geocode_location, find_nearby_hospitals, get_place_details, get_travel_time


# Disease / condition → medical specialty keywords for hospital relevance scoring
DISEASE_SPECIALTY_MAP = {
    # Cardiac
    "cardiac":        ["cardiac", "cardiology", "heart", "cardiovascular"],
    "heart":          ["cardiac", "cardiology", "heart", "cardiovascular"],
    "myocardial":     ["cardiac", "cardiology", "heart", "cardiovascular"],
    "angina":         ["cardiac", "cardiology", "heart", "cardiovascular"],
    "arrhythmia":     ["cardiac", "cardiology", "heart", "cardiovascular"],
    "chest pain":     ["cardiac", "cardiology", "heart", "cardiovascular", "emergency"],
    # Respiratory
    "respiratory":    ["pulmonary", "chest", "lung", "respiratory"],
    "pneumonia":      ["pulmonary", "chest", "lung", "respiratory"],
    "asthma":         ["pulmonary", "chest", "lung", "respiratory", "allergy"],
    "copd":           ["pulmonary", "chest", "lung", "respiratory"],
    "bronchitis":     ["pulmonary", "chest", "lung", "respiratory"],
    "breathless":     ["pulmonary", "chest", "lung", "respiratory", "emergency"],
    "shortness":      ["pulmonary", "chest", "lung", "respiratory", "emergency"],
    # Neurological
    "stroke":         ["neuro", "neurology", "brain", "stroke", "emergency"],
    "seizure":        ["neuro", "neurology", "brain", "epilepsy", "emergency"],
    "migraine":       ["neuro", "neurology", "brain"],
    "meningitis":     ["neuro", "neurology", "brain", "infectious"],
    "paralysis":      ["neuro", "neurology", "brain"],
    # Gastrointestinal
    "gastro":         ["gastro", "digestive", "abdomen", "gi"],
    "appendicitis":   ["surgery", "gastro", "abdomen", "emergency"],
    "pancreatitis":   ["gastro", "surgery", "abdomen"],
    "ulcer":          ["gastro", "digestive"],
    "vomiting":       ["gastro", "general", "medicine"],
    "diarrhea":       ["gastro", "general", "medicine"],
    # Orthopedic / Trauma
    "fracture":       ["ortho", "bone", "trauma", "surgery"],
    "orthopedic":     ["ortho", "bone", "trauma"],
    "arthritis":      ["ortho", "rheumatology", "joint"],
    "sprain":         ["ortho", "bone", "trauma"],
    # Infectious / General
    "dengue":         ["general", "infectious", "medicine", "fever"],
    "malaria":        ["general", "infectious", "medicine", "fever"],
    "typhoid":        ["general", "infectious", "medicine"],
    "covid":          ["general", "infectious", "medicine", "respiratory"],
    "fever":          ["general", "infectious", "medicine"],
    # Metabolic
    "diabetes":       ["endocrine", "general", "medicine", "diabetes"],
    "hypertension":   ["cardiology", "general", "medicine"],
    # Kidney / Urology
    "kidney":         ["nephrology", "urology", "kidney"],
    "renal":          ["nephrology", "urology", "kidney"],
    "uti":            ["urology", "nephrology", "general"],
    # Liver
    "liver":          ["gastro", "hepatology", "liver"],
    "hepatitis":      ["gastro", "hepatology", "liver"],
    # Emergency catch-all
    "emergency":      ["emergency", "trauma", "casualty", "icu"],
    "trauma":         ["trauma", "emergency", "surgery", "casualty"],
}


def _get_specialty_keywords(disease_name: str, diagnosis_text: str) -> List[str]:
    """Return specialty keywords relevant to the top diagnosis disease."""
    keywords = []
    combined = (disease_name + " " + diagnosis_text).lower()

    for key, specialties in DISEASE_SPECIALTY_MAP.items():
        if key in combined:
            keywords.extend(specialties)

    # Always include general multi-specialty terms as a catch-all boost
    keywords.extend(["multispeciality", "multi-specialty", "general hospital", "emergency"])
    return list(set(keywords))


def hospital_finder_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Find nearby hospitals ranked by relevance to the patient's diagnosis.

    Flow:
    1. Resolve user coordinates from state["location"]
    2. Search hospitals in expanding radius (5 → 10 → 20 km)
    3. Score each hospital based on diagnosis specialty + symptom keywords
    4. Return top 3 diagnosis-aligned + 2 nearest (unique, total 5)
    """
    loc = state.get("location") or {}
    lat = loc.get("lat")
    lng = loc.get("lng")

    # Geocode if coordinates are missing
    if lat is None or lng is None:
        text = loc.get("text") or loc.get("formatted") or ""
        if text:
            try:
                geo = geocode_location(text)
                lat = geo.get("lat")
                lng = geo.get("lng")
                if lat is not None and lng is not None:
                    loc.update({"lat": lat, "lng": lng, "formatted": geo.get("formatted")})
            except Exception:
                pass

    if lat is None or lng is None:
        return {"hospitals": [], "hospital_search_meta": {}}

    # Expanding radius search
    radius_steps = [5000, 10000, 20000]
    results = []
    radius_used = None
    for radius in radius_steps:
        try:
            found = find_nearby_hospitals(lat, lng, radius_m=radius)
        except Exception:
            found = []
        if found:
            results = found
            radius_used = radius
            break

    # ── Extract top diagnosis context ──
    top_disease = ""
    diagnosis_text = ""
    try:
        diagnoses = (state.get("diagnosis") or {}).get("diagnoses", [])
        if diagnoses:
            sorted_diag = sorted(
                diagnoses,
                key=lambda d: float(d.get("confidence", 0)),
                reverse=True,
            )
            top = sorted_diag[0]
            top_disease = top.get("disease", "")
            diagnosis_text = f"{top_disease} {top.get('reason', '')}".lower()
    except Exception:
        pass

    symptom_text = " ".join((state.get("patient") or {}).get("symptoms", [])).lower()

    # Build relevance keyword sets
    specialty_keywords = _get_specialty_keywords(top_disease, diagnosis_text)
    diagnosis_keywords = [k for k in diagnosis_text.replace("/", " ").split() if len(k) > 3]
    symptom_keywords   = [k for k in symptom_text.replace("/", " ").split() if len(k) > 3]

    def _haversine_m(lat1, lon1, lat2, lon2):
        R = 6371000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi   = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # ── Score each hospital ──
    hospitals: List[Dict[str, Any]] = []
    for r in results[:40]:
        place_id = r.get("place_id")
        details  = get_place_details(place_id) if place_id else {}

        travel = {}
        try:
            if r.get("lat") is not None and r.get("lng") is not None:
                travel = get_travel_time(lat, lng, r["lat"], r["lng"])
        except Exception:
            pass

        approx_dist = None
        try:
            if r.get("lat") is not None and r.get("lng") is not None:
                approx_dist = _haversine_m(lat, lng, float(r["lat"]), float(r["lng"]))
        except Exception:
            pass

        name_text    = (r.get("name") or "").lower()
        address_text = (details.get("address") or r.get("address") or "").lower()
        combined     = f"{name_text} {address_text}"

        # Relevance scoring
        relevance = 0
        if any(kw in combined for kw in specialty_keywords):
            relevance += 4   # strongest: matches diagnosis specialty
        if any(k in combined for k in diagnosis_keywords):
            relevance += 3   # diagnosis text keyword match
        if any(k in combined for k in symptom_keywords):
            relevance += 2   # symptom keyword match
        if any(kw in combined for kw in ["emergency", "trauma", "casualty", "multispeciality", "multi-specialty"]):
            relevance += 1   # general capability boost

        hospitals.append({
            "name":          r.get("name"),
            "address":       details.get("address") or r.get("address"),
            "phone":         details.get("phone"),
            "distance_m":    travel.get("distance_m") or r.get("distance_m") or approx_dist,
            "travel_time_s": travel.get("duration_s"),
            "place_id":      place_id,
            "relevance_score": relevance,
            "aligned":       False,  # set below for top 3
        })

    # ── Select top 3 aligned + 2 nearest ──
    hospitals_by_relevance = sorted(
        hospitals,
        key=lambda h: (-int(h.get("relevance_score") or 0), float(h.get("distance_m") or 1e12)),
    )
    aligned_pool = [h for h in hospitals_by_relevance if int(h.get("relevance_score") or 0) > 0]
    top_aligned  = aligned_pool[:3]
    for h in top_aligned:
        h["aligned"] = True

    nearest_pool = sorted(hospitals, key=lambda h: float(h.get("distance_m") or 1e12))

    selected, seen = [], set()
    for h in top_aligned:
        pid = h.get("place_id")
        if pid not in seen:
            selected.append(h)
            seen.add(pid)
    for h in nearest_pool:
        if len(selected) >= 5:
            break
        pid = h.get("place_id")
        if pid not in seen:
            selected.append(h)
            seen.add(pid)

    return {
        "hospitals": selected[:5],
        "hospital_search_meta": {
            "radius_used_m":  radius_used,
            "aligned_count":  len(top_aligned),
            "top_diagnosis":  top_disease,
        },
    }
