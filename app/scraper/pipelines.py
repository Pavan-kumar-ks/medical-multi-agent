"""Scrapy item pipelines — clean and deduplicate doctor records."""
import re
from datetime import datetime


class CleanDoctorPipeline:
    """Normalise and sanitise all doctor fields."""

    _PHONE_RE = re.compile(r"[\d\+\-\(\)\s]{7,20}")

    def process_item(self, item, spider):
        # Name — strip titles if duplicated, normalise whitespace
        name = self._clean(item.get("name", ""))
        if name and not name.lower().startswith("dr"):
            name = f"Dr. {name}"
        item["name"] = name

        # Specialty — title-case
        item["specialty"] = self._clean(item.get("specialty", "")).title()

        # Qualifications — normalise commas
        item["qualifications"] = self._clean(item.get("qualifications", ""))

        # Phone — extract only digit patterns, remove tel: prefix
        raw_phone = self._clean(item.get("phone", ""))
        raw_phone = raw_phone.replace("tel:", "").replace("mailto:", "")
        m = self._PHONE_RE.search(raw_phone)
        item["phone"] = m.group().strip() if m else ""

        # Availability — strip excessive whitespace
        item["availability"] = self._clean(item.get("availability", ""))

        # Experience — keep as-is but trim
        item["experience"] = self._clean(item.get("experience", ""))

        # Ensure provenance fields are set
        if not item.get("last_updated"):
            item["last_updated"] = datetime.utcnow().isoformat() + "Z"

        return item

    @staticmethod
    def _clean(text):
        if not text:
            return ""
        return re.sub(r"\s+", " ", str(text)).strip()


class DeduplicatePipeline:
    """Drop duplicate doctors by normalised name within the same hospital."""

    def __init__(self):
        self._seen: set = set()

    def process_item(self, item, spider):
        key = (
            self._norm(item.get("name", "")),
            self._norm(item.get("clinic_hospital", "")),
        )
        if key in self._seen:
            from scrapy.exceptions import DropItem
            raise DropItem(f"Duplicate doctor: {item.get('name')}")
        self._seen.add(key)
        return item

    @staticmethod
    def _norm(text):
        return re.sub(r"\W+", "", (text or "").lower())
