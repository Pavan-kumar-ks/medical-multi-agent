"""Hospital Detail Agent — fetches hospital info and scrapes relevant doctors.

Flow:
1. Use web search to get hospital contact details (phone, website, appointments).
2. Use Scrapy + Playwright scraper to extract structured doctor profiles.
3. Filter doctors by disease-relevant specialty.
4. Fall back to LLM extraction from web snippets if scraper returns nothing.
"""
import json
import logging
from typing import Dict, Any, List, Optional

from app.config import llm_call
from app.tools.web_search import web_search

logger = logging.getLogger(__name__)


# ── Disease → Specialty mapping ───────────────────────────────────────────────

_SPECIALTY_MAP: Dict[str, str] = {
    # Cardiac
    "cardiac":      "cardiologist",   "heart":         "cardiologist",
    "myocardial":   "cardiologist",   "angina":        "cardiologist",
    "arrhythmia":   "cardiologist",   "chest pain":    "cardiologist",
    "hypertension": "cardiologist",
    # Respiratory
    "respiratory":  "pulmonologist",  "asthma":        "pulmonologist",
    "pneumonia":    "pulmonologist",  "copd":          "pulmonologist",
    "bronchitis":   "pulmonologist",  "breathless":    "pulmonologist",
    # Neurological
    "stroke":       "neurologist",    "seizure":       "neurologist",
    "migraine":     "neurologist",    "meningitis":    "neurologist",
    "paralysis":    "neurologist",    "epilepsy":      "neurologist",
    # Gastrointestinal
    "gastro":       "gastroenterologist",
    "appendicitis": "general surgeon",
    "pancreatitis": "gastroenterologist",
    "ulcer":        "gastroenterologist",
    "liver":        "hepatologist",   "hepatitis":     "hepatologist",
    # Orthopaedic
    "fracture":     "orthopedic surgeon",
    "arthritis":    "rheumatologist", "sprain":        "orthopedic surgeon",
    # Metabolic
    "diabetes":     "endocrinologist", "thyroid":      "endocrinologist",
    # Renal / Urology
    "kidney":       "nephrologist",   "renal":         "nephrologist",
    "uti":          "urologist",
    # Oncology
    "cancer":       "oncologist",     "tumor":         "oncologist",
    # Dermatology
    "skin":         "dermatologist",  "rash":          "dermatologist",
    # Ophthalmology
    "eye":          "ophthalmologist", "vision":       "ophthalmologist",
    # ENT
    "ear":          "ENT specialist", "throat":        "ENT specialist",
    # Psychiatry
    "mental":       "psychiatrist",   "depression":    "psychiatrist",
    "anxiety":      "psychiatrist",
    # Gynaecology
    "pregnant":     "gynaecologist",  "obstetric":     "gynaecologist",
    # Paediatrics
    "child":        "paediatrician",  "infant":        "paediatrician",
    # General / Infectious
    "dengue":       "general physician",
    "malaria":      "general physician",
    "typhoid":      "general physician",
    "fever":        "general physician",
}


def disease_to_specialty(disease: str) -> str:
    """Map a disease name to its primary treating medical specialty."""
    d = disease.lower()
    for key, spec in _SPECIALTY_MAP.items():
        if key in d:
            return spec
    return "specialist"


# ── Hospital contact info (web search + LLM extraction) ───────────────────────

def _fetch_hospital_info(
    hospital_name: str, location: str, specialty: str
) -> Dict[str, Any]:
    """Search the web for hospital contact, website, and appointment details."""
    queries = [
        f"{hospital_name} {location} hospital contact number website appointment",
        f"{hospital_name} {location} official website booking",
    ]
    snippets = []
    source_urls = []
    for q in queries:
        results = web_search(q, max_results=3)
        for r in results:
            if r.get("snippet"):
                snippets.append(
                    f"[{r.get('title', '')}]\n{r.get('url', '')}\n{r.get('snippet', '')}"
                )
                source_urls.append(r.get("url", ""))

    snippet_text = "\n\n".join(snippets[:5]) or "No search results found."

    prompt = f"""Extract hospital contact information from these search results.

Hospital: {hospital_name}
Location: {location}

Search Results:
{snippet_text}

Return ONLY valid JSON. Use null when a field is not found.
Do NOT invent phone numbers, URLs, or addresses.

{{
  "hospital_name": "{hospital_name}",
  "website": null,
  "phone_numbers": [],
  "address": null,
  "emergency_number": null,
  "appointment_info": "how to book (phone/online/walk-in)",
  "booking_url": null,
  "departments": [],
  "summary": "1-2 sentences about the hospital"
}}
"""
    response = llm_call(prompt)
    try:
        data = json.loads(response)
    except Exception:
        start, end = response.find("{"), response.rfind("}") + 1
        try:
            data = json.loads(response[start:end]) if start != -1 and end > start else {}
        except Exception:
            data = {}

    data.setdefault("hospital_name", hospital_name)
    data.setdefault("website", source_urls[0] if source_urls else None)
    data.setdefault("phone_numbers", [])
    data.setdefault("appointment_info", f"Contact {hospital_name} directly.")
    data.setdefault("departments", [specialty.title()])
    data.setdefault("summary", f"{hospital_name} provides medical services including {specialty}.")
    return data


# ── Doctor scraping with Scrapy ───────────────────────────────────────────────

def _scrape_doctors(
    hospital_name: str, specialty: str, location: str, website: Optional[str]
) -> List[Dict[str, Any]]:
    """Run the Scrapy spider to extract doctor profiles."""
    try:
        from app.scraper.runner import scrape_doctors
        doctors = scrape_doctors(
            hospital_name=hospital_name,
            specialty=specialty,
            location=location,
            start_url=website,
            timeout=90,
        )
        return doctors
    except Exception as e:
        logger.warning(f"Scrapy runner failed: {e}")
        return []


# ── LLM doctor fallback ───────────────────────────────────────────────────────

def _llm_doctor_fallback(
    hospital_name: str, specialty: str, location: str
) -> List[Dict[str, Any]]:
    """Use web search + LLM to extract doctor info when scraping fails."""
    query = f"{hospital_name} {location} {specialty} doctor specialist name contact"
    results = web_search(query, max_results=5)
    snippets = "\n\n".join(
        f"[{r.get('title','')}]\n{r.get('url','')}\n{r.get('snippet','')}"
        for r in results if r.get("snippet")
    ) or "No results."

    prompt = f"""Extract a list of {specialty} doctors from these search results.

Hospital: {hospital_name}, {location}
Specialty needed: {specialty}

Search Results:
{snippets}

Return ONLY valid JSON array. Include only real doctors found in the results.
Do NOT invent names, phone numbers, or credentials.

[
  {{
    "name": "Dr. Full Name",
    "specialty": "{specialty}",
    "qualifications": "MBBS, MD etc. or null",
    "clinic_hospital": "{hospital_name}",
    "location": "{location}",
    "phone": "phone number or null",
    "appointment_url": "URL or null",
    "availability": "days/hours or null",
    "experience": "X years or null",
    "source_url": "source URL",
    "last_updated": "today"
  }}
]
"""
    response = llm_call(prompt)
    try:
        start = response.find("[")
        end   = response.rfind("]") + 1
        if start != -1 and end > start:
            return json.loads(response[start:end])
    except Exception:
        pass
    return []


# ── Main agent ────────────────────────────────────────────────────────────────

def hospital_detail_agent(
    hospital_name: str,
    disease: str,
    location: str = "",
) -> Dict[str, Any]:
    """Fetch full hospital details and scrape relevant doctors.

    Args:
        hospital_name : selected hospital name
        disease       : diagnosed disease (used to determine specialty)
        location      : city/area for narrowing search

    Returns a dict with hospital info + filtered doctor list.
    """
    specialty = disease_to_specialty(disease)
    logger.info(f"Hospital detail agent: {hospital_name!r}, disease={disease!r}, specialty={specialty!r}")

    # ── Step 1: Hospital contact info ──────────────────────────────────────
    hosp_info = _fetch_hospital_info(hospital_name, location, specialty)
    website   = hosp_info.get("website")

    # ── Step 2: Scrape doctors with Scrapy + Playwright ────────────────────
    print(f"  🕷️  Scraping doctor profiles from {hospital_name}...")
    doctors = _scrape_doctors(hospital_name, specialty, location, website)

    # ── Step 3: LLM fallback if scraper returned nothing ──────────────────
    if not doctors:
        print("  📋  Scraper found no profiles — using web search fallback...")
        doctors = _llm_doctor_fallback(hospital_name, specialty, location)

    # ── Step 4: Filter doctors by specialty relevance ─────────────────────
    specialty_lower = specialty.lower()

    def _is_relevant(doc: Dict) -> bool:
        text = " ".join(filter(None, [
            doc.get("specialty", ""),
            doc.get("qualifications", ""),
            doc.get("name", ""),
        ])).lower()
        return specialty_lower.split()[0] in text or "general" in text

    filtered = [d for d in doctors if _is_relevant(d)]
    if not filtered:
        filtered = doctors  # show all if none match specialty

    # ── Assemble final result ──────────────────────────────────────────────
    return {
        **hosp_info,
        "doctors":   filtered[:8],           # top 8 relevant doctors
        "_specialty": specialty,
        "_doctor_count": len(filtered),
        "_scraper_used": True,
        "_search_used": bool(filtered),
    }
