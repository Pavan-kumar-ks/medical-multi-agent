"""Fixed doctor schema — every field is extracted for every doctor record."""
import scrapy


class DoctorItem(scrapy.Item):
    # ── Identity ──────────────────────────────────────────────────────────────
    name             = scrapy.Field()   # Full name, e.g. "Dr. Vivek Jawali"
    specialty        = scrapy.Field()   # Medical specialty, e.g. "Cardiologist"
    qualifications   = scrapy.Field()   # Degrees/credentials, e.g. "MBBS, MD, DM"

    # ── Location ──────────────────────────────────────────────────────────────
    clinic_hospital  = scrapy.Field()   # Hospital/clinic name
    location         = scrapy.Field()   # City / address of the hospital

    # ── Contact ───────────────────────────────────────────────────────────────
    phone            = scrapy.Field()   # Direct or reception phone number
    appointment_url  = scrapy.Field()   # Online booking URL (if found)

    # ── Availability ─────────────────────────────────────────────────────────
    availability     = scrapy.Field()   # Consulting days/hours, e.g. "Mon–Fri 10–1"
    experience       = scrapy.Field()   # Years of experience or seniority note

    # ── Provenance ────────────────────────────────────────────────────────────
    source_url       = scrapy.Field()   # URL the record was scraped from
    last_updated     = scrapy.Field()   # ISO timestamp of scrape time
