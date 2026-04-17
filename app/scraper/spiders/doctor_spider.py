"""Doctor Spider — crawls hospital websites to extract doctor profiles.

Strategy:
1. Start at the URL passed by the runner (hospital website or doctors page).
2. Try static HTML extraction (fast path).
3. If too few results, retry with Playwright for JS-rendered pages.
4. Follow internal links matching doctor/team/specialist patterns.
5. Extract against multiple CSS selector strategies + JSON-LD fallback.
6. Filter extracted doctors by target specialty when provided.
"""
import re
import json
import sys
import os
from datetime import datetime
from typing import List, Iterator, Optional

import scrapy
from scrapy_playwright.page import PageMethod

# Allow importing items from parent package when run via subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from items import DoctorItem


# ── Selector banks ────────────────────────────────────────────────────────────

# Containers that hold a single doctor's information
_CARD_SELECTORS = [
    ".doctor-card", ".physician-card", ".doctor-profile", ".doctor-item",
    "[class*='doctor-card']", "[class*='physician-card']",
    "[class*='doctor-profile']", "[class*='doctor-item']",
    "[class*='specialist-card']", "[class*='specialist-item']",
    ".team-member", ".team-card", ".staff-card", ".provider-card",
    "[class*='team-member']", "[class*='staff-card']",
    "li.doctor", "div.doctor", "article.doctor",
]

# Name within a card
_NAME_SELECTORS = [
    ".doctor-name", ".physician-name", ".name", ".doctor-title",
    "[class*='doctor-name']", "[class*='physician-name']",
    "[itemprop='name']", "h1", "h2", "h3", "h4", "strong.name",
]

# Specialty within a card
_SPECIALTY_SELECTORS = [
    ".specialty", ".specialization", ".designation", ".department",
    ".doctor-specialty", ".physician-specialty",
    "[class*='specialty']", "[class*='specializ']", "[class*='department']",
    "[class*='designation']", "p.specialty", "span.specialty",
    "[itemprop='medicalSpecialty']",
]

# Qualifications
_QUAL_SELECTORS = [
    ".qualifications", ".credentials", ".education", ".degree",
    "[class*='qualif']", "[class*='credential']", "[class*='degree']",
    ".mbbs", ".md", ".doctor-qualifications",
]

# Availability / timing
_AVAIL_SELECTORS = [
    ".availability", ".timing", ".schedule", ".consultation-time",
    "[class*='availab']", "[class*='timing']", "[class*='schedule']",
    ".hours", ".days",
]

# Experience
_EXP_SELECTORS = [
    ".experience", ".years-experience", "[class*='experience']",
    ".exp", "[class*='exp']",
]

# Internal link patterns that indicate a doctors/team page
_DOCTOR_LINK_PATTERNS = [
    "doctor", "physician", "specialist", "consultant", "team",
    "our-team", "find-a-doctor", "medical-staff", "provider",
    "expert", "surgeon", "faculty",
]


# ── Spider ────────────────────────────────────────────────────────────────────

class DoctorSpider(scrapy.Spider):
    name = "doctor_spider"

    custom_settings = {
        "PLAYWRIGHT_BROWSER_TYPE": "chromium",
        "DOWNLOAD_HANDLERS": {
            "http":  "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
        "PLAYWRIGHT_LAUNCH_OPTIONS": {
            "headless": True,
            "args": ["--no-sandbox", "--disable-dev-shm-usage"],
        },
        "PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT": 20_000,
    }

    def __init__(
        self,
        start_url: str = "",
        hospital_name: str = "",
        specialty: str = "",
        location: str = "",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.start_urls      = [start_url] if start_url else []
        self.hospital_name   = hospital_name
        self.target_specialty = specialty.lower().strip()
        self.hospital_location = location
        self._visited: set   = set()

    # ── Entry ──────────────────────────────────────────────────────────────

    def start_requests(self):
        for url in self.start_urls:
            # First attempt: fast static request
            yield scrapy.Request(
                url,
                callback=self._parse_static_first,
                errback=self._playwright_fallback_errback,
                meta={"original_url": url},
                dont_filter=False,
            )

    # ── Parse helpers ──────────────────────────────────────────────────────

    def _parse_static_first(self, response):
        """Try static extraction; escalate to Playwright if page is JS-heavy."""
        doctors = list(self._extract_doctors(response))
        if doctors:
            yield from doctors
            yield from self._follow_doctor_links(response, use_playwright=True)
        else:
            # Retry with Playwright
            yield self._playwright_request(response.url, self._parse_playwright)

    def _parse_playwright(self, response):
        """Parse after Playwright JS rendering."""
        yield from self._extract_doctors(response)
        yield from self._follow_doctor_links(response, use_playwright=True)

    def _playwright_fallback_errback(self, failure):
        """On static fetch error, try Playwright."""
        url = failure.request.meta.get("original_url", failure.request.url)
        yield self._playwright_request(url, self._parse_playwright)

    def _playwright_request(self, url: str, callback) -> scrapy.Request:
        return scrapy.Request(
            url,
            callback=callback,
            meta={
                "playwright": True,
                "playwright_include_page": False,
                "playwright_page_methods": [
                    PageMethod("wait_for_load_state", "networkidle"),
                ],
                "playwright_context_kwargs": {
                    "ignore_https_errors": True,
                },
            },
            dont_filter=True,
        )

    # ── Doctor extraction ──────────────────────────────────────────────────

    def _extract_doctors(self, response) -> Iterator[DoctorItem]:
        """Try all extraction strategies in order."""
        # Strategy 1: Card-based CSS extraction
        found_any = False
        for card_sel in _CARD_SELECTORS:
            cards = response.css(card_sel)
            if not cards:
                continue
            for card in cards:
                item = self._item_from_card(card, response.url)
                if item and item.get("name"):
                    if self._matches_specialty(item):
                        found_any = True
                        yield item
            if found_any:
                return

        # Strategy 2: JSON-LD structured data
        for item in self._extract_json_ld(response):
            if self._matches_specialty(item):
                yield item
                found_any = True
        if found_any:
            return

        # Strategy 3: hCard / microdata
        yield from self._extract_microdata(response)

    def _item_from_card(self, card, source_url: str) -> Optional[DoctorItem]:
        """Extract a DoctorItem from a card CSS element."""
        name      = self._first_text(card, _NAME_SELECTORS)
        specialty = self._first_text(card, _SPECIALTY_SELECTORS)
        qual      = self._first_text(card, _QUAL_SELECTORS)
        avail     = self._first_text(card, _AVAIL_SELECTORS)
        exp       = self._first_text(card, _EXP_SELECTORS)

        # Phone — prefer tel: href, fall back to text
        phone = ""
        tel_href = card.css("a[href^='tel:']::attr(href)").get("")
        if tel_href:
            phone = tel_href.replace("tel:", "").strip()
        else:
            phone = self._first_text(card, [".phone", "[class*='phone']", ".contact-phone"])

        # Appointment URL
        appt_url = ""
        for href in card.css("a::attr(href)").getall():
            if any(k in href.lower() for k in ["book", "appoint", "consult", "schedule"]):
                appt_url = href
                break

        if not name:
            return None

        return DoctorItem(
            name=name,
            specialty=specialty or self.target_specialty.title(),
            qualifications=qual,
            clinic_hospital=self.hospital_name,
            location=self.hospital_location,
            phone=phone,
            appointment_url=appt_url,
            availability=avail,
            experience=exp,
            source_url=source_url,
            last_updated=datetime.utcnow().isoformat() + "Z",
        )

    def _extract_json_ld(self, response) -> Iterator[DoctorItem]:
        """Extract from JSON-LD <script> blocks."""
        for raw in response.css('script[type="application/ld+json"]::text').getall():
            try:
                data = json.loads(raw)
            except Exception:
                continue

            # Handle single object or @graph list
            records = data if isinstance(data, list) else data.get("@graph", [data])

            for record in records:
                rtype = record.get("@type", "")
                if rtype in ("Physician", "MedicalOrganization", "Person", "MedicalBusiness"):
                    name = record.get("name", "")
                    if not name:
                        continue
                    addr = record.get("address", {})
                    location = (
                        addr.get("streetAddress", "") + " " + addr.get("addressLocality", "")
                        if isinstance(addr, dict)
                        else str(addr)
                    ).strip() or self.hospital_location

                    yield DoctorItem(
                        name=name,
                        specialty=record.get("medicalSpecialty", self.target_specialty.title()),
                        qualifications="",
                        clinic_hospital=record.get("memberOf", {}).get("name", self.hospital_name)
                        if isinstance(record.get("memberOf"), dict) else self.hospital_name,
                        location=location,
                        phone=str(record.get("telephone", "")),
                        appointment_url=record.get("url", ""),
                        availability="",
                        experience="",
                        source_url=response.url,
                        last_updated=datetime.utcnow().isoformat() + "Z",
                    )

    def _extract_microdata(self, response) -> Iterator[DoctorItem]:
        """Extract from HTML microdata (itemprop attributes)."""
        for scope in response.css("[itemtype*='schema.org/Physician'], [itemtype*='schema.org/Person']"):
            name = scope.css("[itemprop='name']::text").get("").strip()
            if not name:
                continue
            yield DoctorItem(
                name=name,
                specialty=scope.css("[itemprop='medicalSpecialty']::text").get("").strip()
                         or self.target_specialty.title(),
                qualifications=scope.css("[itemprop='hasCredential']::text").get("").strip(),
                clinic_hospital=self.hospital_name,
                location=self.hospital_location,
                phone=scope.css("[itemprop='telephone']::text").get("").strip(),
                appointment_url=scope.css("[itemprop='url']::attr(href)").get(""),
                availability="",
                experience="",
                source_url=response.url,
                last_updated=datetime.utcnow().isoformat() + "Z",
            )

    # ── Link following ─────────────────────────────────────────────────────

    def _follow_doctor_links(self, response, use_playwright: bool = True):
        """Yield requests for internal links that likely lead to doctor pages."""
        for href in response.css("a::attr(href)").getall():
            full_url = response.urljoin(href)
            # Stay on same domain
            if response.url.split("/")[2] not in full_url:
                continue
            if full_url in self._visited:
                continue
            href_lower = href.lower()
            if any(p in href_lower for p in _DOCTOR_LINK_PATTERNS):
                self._visited.add(full_url)
                if use_playwright:
                    yield self._playwright_request(full_url, self._parse_playwright)
                else:
                    yield scrapy.Request(full_url, callback=self._parse_static_first)

    # ── Utility ────────────────────────────────────────────────────────────

    def _first_text(self, element, selectors: List[str]) -> str:
        for sel in selectors:
            text = element.css(f"{sel}::text").get("").strip()
            if text and 2 < len(text) < 200:
                return text
            # Also try ::attr(content) for meta-like tags
            text = element.css(f"{sel}::attr(content)").get("").strip()
            if text and 2 < len(text) < 200:
                return text
        return ""

    def _matches_specialty(self, item: DoctorItem) -> bool:
        """Return True if this doctor matches the target specialty (or no filter set)."""
        if not self.target_specialty:
            return True
        combined = " ".join(filter(None, [
            item.get("specialty", ""),
            item.get("qualifications", ""),
            item.get("name", ""),
        ])).lower()
        # Accept if target specialty or common related terms appear
        return self.target_specialty.split()[0] in combined or "general" in combined
