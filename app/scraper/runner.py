"""Doctor scraper — requests + BeautifulSoup + Playwright + LLM extraction.

No Scrapy subprocess. Works on Windows and Linux without Twisted/reactor issues.

Strategy per hospital:
1. Web-search for the hospital's official doctors / specialists page URL.
2. Fetch HTML with requests (fast static path).
3. Parse with BeautifulSoup: card-selectors → JSON-LD → microdata.
4. Also run LLM extraction on cleaned page text (works regardless of HTML structure).
5. If total < 2 and page looks JS-heavy, retry with Playwright.
6. Follow internal "doctor/team/specialist" links one level deep.
7. Clean + deduplicate results.
"""
import re
import sys
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests

logger = logging.getLogger(__name__)

_SCRAPER_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRAPER_DIR.parent))   # makes app.* importable

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# ── CSS selector banks (broad coverage for Indian hospital websites) ──────────

_CARD_SELECTORS = [
    ".doctor-card", ".physician-card", ".doctor-profile", ".doctor-item",
    "[class*='doctor-card']", "[class*='physician-card']",
    "[class*='doctor-profile']", "[class*='doctor-item']",
    "[class*='specialist-card']", "[class*='specialist-item']",
    ".team-member", ".team-card", ".staff-card", ".provider-card",
    "[class*='team-member']", "[class*='staff-card']",
    "li.doctor", "div.doctor", "article.doctor",
    ".our-doctors .item", ".doctors-list .item",
]

_NAME_SELECTORS = [
    ".doctor-name", ".physician-name", ".name", ".doctor-title",
    "[class*='doctor-name']", "[class*='physician-name']",
    "[itemprop='name']", "h2", "h3", "h4", "strong.name",
]
_SPEC_SELECTORS = [
    ".specialty", ".specialization", ".designation", ".department",
    "[class*='specialty']", "[class*='speciali']", "[class*='department']",
    "[class*='designation']", "p.specialty", "span.specialty",
    "[itemprop='medicalSpecialty']",
]
_QUAL_SELECTORS = [
    ".qualifications", ".credentials", ".education", ".degree",
    "[class*='qualif']", "[class*='credential']", "[class*='degree']",
]
_AVAIL_SELECTORS = [
    ".availability", ".timing", ".schedule", ".consultation-time",
    "[class*='availab']", "[class*='timing']", "[class*='schedule']",
]
_EXP_SELECTORS = [".experience", "[class*='experience']", ".exp"]

_DOCTOR_LINK_PATTERNS = [
    "doctor", "physician", "specialist", "consultant", "team",
    "our-team", "find-a-doctor", "medical-staff", "provider",
    "expert", "surgeon", "faculty", "specialists",
]


# ── HTML fetch helpers ────────────────────────────────────────────────────────

def _fetch_html_static(url: str, timeout: int = 20) -> str:
    """Fetch a URL with requests and return raw HTML."""
    for attempt in range(2):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=timeout, allow_redirects=True)
            if resp.status_code == 200:
                return resp.text
            logger.warning(f"HTTP {resp.status_code} for {url}")
        except Exception as e:
            if attempt == 0:
                time.sleep(1)
            else:
                logger.warning(f"Static fetch failed for {url}: {e}")
    return ""


def _fetch_html_playwright(url: str, timeout: int = 45) -> str:
    """Fetch a JS-rendered page with Playwright (sync API). Returns '' if unavailable."""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            try:
                page = browser.new_page()
                page.set_default_navigation_timeout(timeout * 1000)
                page.goto(url)
                page.wait_for_load_state("networkidle", timeout=timeout * 1000)
                html = page.content()
                return html
            finally:
                browser.close()
    except ImportError:
        logger.debug("Playwright not installed — skipping JS render.")
    except Exception as e:
        logger.warning(f"Playwright fetch failed for {url}: {e}")
    return ""


def _html_to_text(html: str, max_chars: int = 8000) -> str:
    """Strip HTML tags and return cleaned, readable text for LLM extraction."""
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "head",
                          "header", "aside", "noscript", "iframe"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    except ImportError:
        text = re.sub(r"<[^>]+>", " ", html)

    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 2]
    return "\n".join(lines)[:max_chars]


def _looks_js_heavy(html: str) -> bool:
    """Heuristic: few visible words but many script tags → JS-rendered."""
    scripts = html.count("<script")
    visible = len(re.sub(r"<[^>]+>", "", html).split())
    return scripts > 10 and visible < 300


# ── Structured extraction (BeautifulSoup) ────────────────────────────────────

def _first_text(element, selectors: list) -> str:
    for sel in selectors:
        try:
            el = element.select_one(sel)
            if el:
                t = el.get_text(" ", strip=True)
                if 2 < len(t) < 200:
                    return t
        except Exception:
            pass
    return ""


def _extract_cards(soup, hospital_name: str, specialty: str,
                   location: str, source_url: str) -> List[Dict]:
    """Card-based CSS extraction — tries each selector bank in order."""
    for sel in _CARD_SELECTORS:
        try:
            cards = soup.select(sel)
        except Exception:
            continue
        if not cards:
            continue
        doctors = []
        for card in cards:
            name = _first_text(card, _NAME_SELECTORS)
            if not name:
                continue
            spec  = _first_text(card, _SPEC_SELECTORS) or specialty
            qual  = _first_text(card, _QUAL_SELECTORS)
            avail = _first_text(card, _AVAIL_SELECTORS)
            exp   = _first_text(card, _EXP_SELECTORS)

            phone = ""
            tel = card.select_one("a[href^='tel:']")
            if tel:
                phone = tel["href"].replace("tel:", "").strip()
            else:
                phone = _first_text(card, [".phone", "[class*='phone']"])

            appt_url = ""
            for a in card.select("a[href]"):
                href = a.get("href", "")
                if any(k in href.lower() for k in ["book", "appoint", "consult", "schedule"]):
                    appt_url = href
                    break

            doctors.append({
                "name": name,
                "specialty": spec,
                "qualifications": qual,
                "clinic_hospital": hospital_name,
                "location": location,
                "phone": phone,
                "appointment_url": appt_url,
                "availability": avail,
                "experience": exp,
                "source_url": source_url,
                "last_updated": datetime.utcnow().isoformat() + "Z",
            })
        if doctors:
            return doctors
    return []


def _extract_json_ld(soup, hospital_name: str, specialty: str,
                     location: str, source_url: str) -> List[Dict]:
    """Extract from JSON-LD <script type='application/ld+json'> blocks."""
    doctors = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except Exception:
            continue
        records = data if isinstance(data, list) else data.get("@graph", [data])
        for rec in records:
            rtype = rec.get("@type", "")
            if rtype not in ("Physician", "Person", "MedicalBusiness", "MedicalOrganization"):
                continue
            name = rec.get("name", "")
            if not name or len(name) > 80:
                continue
            addr = rec.get("address", {})
            loc = (
                (addr.get("streetAddress", "") + " " + addr.get("addressLocality", "")).strip()
                if isinstance(addr, dict) else str(addr)
            ) or location
            doctors.append({
                "name": name,
                "specialty": rec.get("medicalSpecialty", specialty),
                "qualifications": "",
                "clinic_hospital": hospital_name,
                "location": loc,
                "phone": str(rec.get("telephone", "") or ""),
                "appointment_url": rec.get("url", ""),
                "availability": "",
                "experience": "",
                "source_url": source_url,
                "last_updated": datetime.utcnow().isoformat() + "Z",
            })
    return doctors


def _extract_bs4(html: str, hospital_name: str, specialty: str,
                 location: str, source_url: str) -> List[Dict]:
    """Run all structured BS4 strategies: cards → JSON-LD."""
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:
        return []
    soup = BeautifulSoup(html, "lxml")
    doctors = _extract_cards(soup, hospital_name, specialty, location, source_url)
    if not doctors:
        doctors = _extract_json_ld(soup, hospital_name, specialty, location, source_url)
    return doctors


# ── Doctor-link discovery ─────────────────────────────────────────────────────

def _find_doctor_links(html: str, base_url: str) -> List[str]:
    """Return internal URLs that look like doctor/team pages."""
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(html, "lxml")
        hrefs = [a.get("href", "") for a in soup.find_all("a", href=True)]
    except ImportError:
        hrefs = re.findall(r'href=["\']([^"\']+)["\']', html)

    domain = ""
    m = re.match(r"(https?://[^/]+)", base_url)
    if m:
        domain = m.group(1)

    seen, links = set(), []
    for href in hrefs:
        if href.startswith("//"):
            href = "https:" + href
        elif href.startswith("/"):
            href = domain + href
        elif not href.startswith("http"):
            continue
        if domain and domain not in href:
            continue
        href_lower = href.lower()
        if any(p in href_lower for p in _DOCTOR_LINK_PATTERNS):
            if href not in seen and href != base_url:
                seen.add(href)
                links.append(href)
    return links[:5]


# ── LLM extraction from page text ────────────────────────────────────────────

def _llm_extract_from_page(page_text: str, hospital_name: str, specialty: str,
                            location: str, source_url: str) -> List[Dict]:
    """Pass cleaned page text to the LLM and extract structured doctor profiles."""
    if not page_text.strip() or len(page_text) < 80:
        return []
    try:
        from config import llm_call  # type: ignore
    except ImportError:
        try:
            from app.config import llm_call  # type: ignore
        except ImportError:
            return []

    today = datetime.utcnow().strftime("%Y-%m-%d")
    prompt = f"""Extract all doctor profiles from this hospital webpage text.

Hospital: {hospital_name}
Location: {location}
Target specialty: {specialty}

Webpage text:
---
{page_text}
---

Instructions:
- Extract EVERY doctor / physician / consultant / surgeon mentioned.
- Only use information actually present in the text above.
- Do NOT invent names, phone numbers, or credentials.
- For any missing field use null.
- Return ONLY a valid JSON array — no explanation, no markdown fences.

[
  {{
    "name": "Dr. Full Name",
    "specialty": "their specialty or null",
    "qualifications": "MBBS / MS / MD etc. or null",
    "clinic_hospital": "{hospital_name}",
    "location": "{location}",
    "phone": "phone number or null",
    "appointment_url": "booking URL or null",
    "availability": "e.g. Mon-Fri 9am-5pm or null",
    "experience": "e.g. 12 years or null",
    "source_url": "{source_url}",
    "last_updated": "{today}"
  }}
]

Return [] if no doctor names appear in the text.
"""
    response = llm_call(prompt)
    try:
        start = response.find("[")
        end   = response.rfind("]") + 1
        if start != -1 and end > start:
            parsed = json.loads(response[start:end])
            return [d for d in parsed if isinstance(d, dict) and d.get("name")]
    except Exception:
        pass
    return []


# ── Cleaning & deduplication ──────────────────────────────────────────────────

_PHONE_RE = re.compile(r"[\d\+\-\(\)\s]{7,20}")


def _clean_doctor(doc: Dict) -> Dict:
    name = re.sub(r"\s+", " ", str(doc.get("name", "") or "")).strip()
    if name and not name.lower().startswith("dr"):
        name = "Dr. " + name
    doc["name"] = name

    doc["specialty"] = str(doc.get("specialty") or "").title().strip()

    raw_phone = str(doc.get("phone", "") or "").replace("tel:", "").replace("mailto:", "")
    m = _PHONE_RE.search(raw_phone)
    doc["phone"] = m.group().strip() if m else ""

    for field in ("qualifications", "availability", "experience", "clinic_hospital", "location"):
        doc[field] = re.sub(r"\s+", " ", str(doc.get(field, "") or "")).strip()

    if not doc.get("last_updated"):
        doc["last_updated"] = datetime.utcnow().isoformat() + "Z"
    return doc


def _deduplicate(doctors: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for d in doctors:
        key = re.sub(r"\W+", "", (d.get("name", "") or "").lower())
        if key and key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _is_valid(doc: Dict) -> bool:
    name = (doc.get("name", "") or "").strip()
    if not name or len(name) < 5:
        return False
    junk = {"dr.", "doctor", "physician", "specialist", "consultant"}
    if name.lower().rstrip(".") in junk:
        return False
    return True


# ── URL discovery ─────────────────────────────────────────────────────────────

def _find_hospital_url(hospital_name: str, specialty: str, location: str) -> Optional[str]:
    """Web-search for the hospital's doctors-page URL."""
    try:
        from tools.web_search import web_search  # type: ignore
    except ImportError:
        try:
            from app.tools.web_search import web_search  # type: ignore
        except ImportError:
            return None

    queries = [
        f"{hospital_name} {location} {specialty} doctors specialists",
        f"{hospital_name} {location} official website",
        f'"{hospital_name}" {location}',
    ]
    for query in queries:
        try:
            results = web_search(query, max_results=5)
            for r in results:
                url = r.get("url", "")
                if url and any(k in url.lower() for k in
                               ["doctor", "specialist", "physician", "team", "expert"]):
                    return url
            for r in results:
                url = r.get("url", "")
                if url and url.startswith("http"):
                    return url
        except Exception:
            pass
    return None


# ── Core public API ───────────────────────────────────────────────────────────

def scrape_doctors(
    hospital_name: str,
    specialty: str,
    location: str = "",
    start_url: Optional[str] = None,
    timeout: int = 90,
) -> List[Dict[str, Any]]:
    """Scrape doctor profiles for a hospital.

    Returns a cleaned, deduplicated list of doctor dicts.
    """
    # ── Step 1: Find the best starting URL ────────────────────────────────
    url = start_url or _find_hospital_url(hospital_name, specialty, location)
    if not url:
        logger.warning("Could not find a hospital URL to scrape.")
        return []
    logger.info(f"Scraping doctors: {hospital_name!r} → {url}")

    all_doctors: List[Dict] = []

    # ── Step 2: Static fetch ──────────────────────────────────────────────
    static_html = _fetch_html_static(url)
    if static_html:
        # Structured (CSS/JSON-LD) extraction
        structured = _extract_bs4(static_html, hospital_name, specialty, location, url)
        all_doctors.extend(structured)

        # LLM extraction from page text (catches what CSS misses)
        page_text = _html_to_text(static_html)
        llm_docs  = _llm_extract_from_page(page_text, hospital_name, specialty, location, url)
        # Merge without duplicating
        known_names = {d.get("name", "").lower() for d in all_doctors}
        for d in llm_docs:
            if d.get("name", "").lower() not in known_names:
                all_doctors.append(d)
                known_names.add(d.get("name", "").lower())

        # Follow internal doctor/team links (one level deep)
        if len(all_doctors) < 3:
            for link in _find_doctor_links(static_html, url)[:3]:
                sub_html = _fetch_html_static(link)
                if not sub_html:
                    continue
                sub_structured = _extract_bs4(sub_html, hospital_name, specialty, location, link)
                sub_text = _html_to_text(sub_html)
                sub_llm  = _llm_extract_from_page(sub_text, hospital_name, specialty, location, link)
                for d in sub_structured + sub_llm:
                    if d.get("name", "").lower() not in known_names:
                        all_doctors.append(d)
                        known_names.add(d.get("name", "").lower())

    # ── Step 3: Playwright fallback if page is JS-heavy ───────────────────
    if len(all_doctors) < 2:
        js_needed = (static_html and _looks_js_heavy(static_html)) or not static_html
        if js_needed:
            logger.info("Retrying with Playwright for JS-rendered page.")
            pw_html = _fetch_html_playwright(url)
            if pw_html:
                structured = _extract_bs4(pw_html, hospital_name, specialty, location, url)
                page_text  = _html_to_text(pw_html)
                llm_docs   = _llm_extract_from_page(page_text, hospital_name, specialty, location, url)
                known_names = {d.get("name", "").lower() for d in all_doctors}
                for d in structured + llm_docs:
                    if d.get("name", "").lower() not in known_names:
                        all_doctors.append(d)

    # ── Step 4: Clean, deduplicate, validate ──────────────────────────────
    cleaned = [_clean_doctor(d) for d in all_doctors]
    deduped = _deduplicate(cleaned)
    valid   = [d for d in deduped if _is_valid(d)]

    logger.info(f"Found {len(valid)} doctor(s) for {hospital_name!r}")
    return valid
