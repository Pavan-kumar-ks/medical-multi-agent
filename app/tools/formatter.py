"""Format medical assistant responses into a clean, attractive terminal output."""
from typing import Any, Dict, List, Optional


# ── Helpers ──────────────────────────────────────────────────────────────────

def _confidence_bar(value: float, width: int = 15) -> str:
    """Visual confidence bar using block characters."""
    filled = int(round(value * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _fmt_distance(distance_m) -> str:
    if not isinstance(distance_m, (int, float)):
        return ""
    return f"{distance_m / 1000:.1f} km"


def _fmt_travel(duration_s) -> str:
    if not isinstance(duration_s, (int, float)):
        return ""
    mins = int(round(duration_s / 60))
    return f"~{mins} min"


def _wrap_text(text: str, width: int, indent: str = "           ") -> List[str]:
    """Word-wrap text to width, with indent on continuation lines."""
    words = text.split()
    lines, buf = [], []
    for word in words:
        if sum(len(x) + 1 for x in buf) + len(word) > width:
            lines.append(indent + " ".join(buf))
            buf = [word]
        else:
            buf.append(word)
    if buf:
        lines.append(indent + " ".join(buf))
    return lines


# ── Main formatter ────────────────────────────────────────────────────────────

def format_medical_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Convert raw graph output into a structured + attractive display.

    Short-circuits to a Q&A format when the response is a follow-up answer.
    """
    W    = 56
    DSEП = "  " + "═" * (W - 4)
    SEP  = "  " + "─" * (W - 4)

    # ── Follow-up short-circuit ───────────────────────────────────────────
    followup_answer = response.get("followup_answer")
    if followup_answer:
        lines = [
            "",
            f"{'💬  Follow-up Answer':^{W}}",
            DSEП,
            "",
        ]
        # Word-wrap the answer into 50-char lines
        for paragraph in followup_answer.split("\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                lines.append("")
                continue
            words, buf = paragraph.split(), []
            for word in words:
                if sum(len(x) + 1 for x in buf) + len(word) > 50:
                    lines.append("  " + " ".join(buf))
                    buf = [word]
                else:
                    buf.append(word)
            if buf:
                lines.append("  " + " ".join(buf))
        lines += ["", DSEП, "  ⚕  Always consult a qualified doctor for personalised advice.", DSEП, ""]
        return {
            "followup_answer": followup_answer,
            "pretty_text": "\n".join(lines),
            "raw": response,
        }

    patient = response.get("patient") or {}

    # ── Extract fields ──
    risks: List[str] = []
    if isinstance(response.get("risks"), dict):
        risks = response["risks"].get("risks", [])
    elif isinstance(response.get("risks"), list):
        risks = response["risks"]

    tests: List[Any] = []
    raw_tests = response.get("tests") or {}
    if isinstance(raw_tests, dict):
        tests = raw_tests.get("tests", [])
    elif isinstance(raw_tests, list):
        tests = raw_tests

    remedy_steps: List[str] = []
    raw_remedy = response.get("remedy") or {}
    if isinstance(raw_remedy, dict):
        remedy_steps = raw_remedy.get("remedy_steps", [])

    emergency  = bool(response.get("is_emergency") or False)
    hospitals  = response.get("hospitals") or []
    hosp_meta  = response.get("hospital_search_meta") or {}

    diagnoses: List[Dict] = []
    raw_diag = response.get("diagnosis")
    if isinstance(raw_diag, dict):
        diagnoses = raw_diag.get("diagnoses", [])
    if diagnoses:
        diagnoses = sorted(diagnoses, key=lambda d: float(d.get("confidence", 0)), reverse=True)

    symptoms  = patient.get("symptoms") or []
    age       = patient.get("age")
    gender    = patient.get("gender")
    duration  = patient.get("duration_days")

    # (Layout constants W / SEP / DSEП defined at function top)

    lines: List[str] = []

    # ════ Header ════
    lines.append("")
    if emergency:
        lines.append(f"{'🚨  EMERGENCY — SEEK IMMEDIATE HELP  🚨':^{W}}")
    else:
        lines.append(f"{'🏥   Medical Assessment Report   🏥':^{W}}")
    lines.append(DSEП)

    # ── Patient snapshot ──
    info_parts = []
    if age:       info_parts.append(f"Age: {age}")
    if gender:    info_parts.append(f"Gender: {gender.title()}")
    if duration:  info_parts.append(f"Duration: {duration} day(s)")
    if info_parts:
        lines.append("  👤  " + "   |   ".join(info_parts))
    if symptoms:
        lines.append(f"  🤒  Symptoms  :  {', '.join(symptoms)}")
    lines.append(SEP)

    # ════ Diagnosis ════
    if diagnoses:
        lines.append("  📋  DIAGNOSIS")
        lines.append(SEP)
        for i, d in enumerate(diagnoses[:3]):
            conf  = float(d.get("confidence", 0))
            bar   = _confidence_bar(conf)
            label = "  ⭐ Best Match" if i == 0 else f"  #{i + 1}          "
            lines.append(f"{label}  {d.get('disease', 'Unknown')}")
            lines.append(f"            Confidence : [{bar}] {conf:.0%}")
            reason = (d.get("reason") or "").strip()
            if reason:
                for wrapped_line in _wrap_text(reason, 44, "            "):
                    lines.append(wrapped_line)
            lines.append("")

    # ════ Emergency First-Aid ════
    if emergency and remedy_steps:
        lines.append(SEP)
        lines.append("  🚨  IMMEDIATE ACTIONS")
        lines.append(SEP)
        for step in remedy_steps[:5]:
            lines.append(f"  ▸  {step}")
        lines.append("")

    # ════ Hospitals ════
    if hospitals:
        lines.append(SEP)
        top_dx    = hosp_meta.get("top_diagnosis", "")
        aligned   = [h for h in hospitals if h.get("aligned")]
        others    = [h for h in hospitals if not h.get("aligned")]

        lines.append("  🏨  NEARBY HOSPITALS")
        if top_dx:
            lines.append(f"      Matched for : {top_dx}")
        lines.append(SEP)

        def _render_hospital(h: Dict, tag: str = "🏥"):
            name   = h.get("name") or "Hospital"
            dist   = _fmt_distance(h.get("distance_m"))
            travel = _fmt_travel(h.get("travel_time_s"))
            addr   = h.get("address") or ""
            phone  = h.get("phone") or ""

            meta = ""
            if dist and travel:
                meta = f"  ({dist}, {travel})"
            elif dist:
                meta = f"  ({dist})"

            lines.append(f"  {tag}  {name}{meta}")
            if addr:
                lines.append(f"       📍  {addr}")
            if phone:
                lines.append(f"       📞  {phone}")
            lines.append("")

        if aligned:
            lines.append("  ✅  Best match for your diagnosis:")
            lines.append("")
            for h in aligned:
                _render_hospital(h, "🏥")

        if others:
            lines.append("  📍  Other nearby hospitals:")
            lines.append("")
            for h in others:
                _render_hospital(h, "🏥")

    elif hosp_meta.get("radius_used_m"):
        lines.append(SEP)
        lines.append("  🏨  HOSPITALS")
        lines.append(SEP)
        lines.append("       No hospitals found in the search area.")
        lines.append("")

    # ════ Panel Review Summary ════
    panel_decision = response.get("panel_decision") or {}
    if panel_decision:
        lines.append(SEP)
        lines.append("  🧠  PANEL REVIEW")
        lines.append(SEP)

        # Conflicts detected
        n_conflicts = panel_decision.get("conflict_count", 0)
        uncertainty = panel_decision.get("uncertainty_flag", False)
        if n_conflicts == 0:
            lines.append("  ✅  Panel reached full consensus — no conflicts.")
        else:
            severity_icon = "🔴" if n_conflicts >= 3 else "🟡"
            lines.append(f"  {severity_icon}  {n_conflicts} conflict(s) detected and resolved.")

        # Panel summary line
        panel_summary = panel_decision.get("panel_summary", "")
        if panel_summary:
            for l in _wrap_text(panel_summary, 46, "      "):
                lines.append(l)
        lines.append("")

        # Why conflict / why winner
        conflict_reason = panel_decision.get("conflict_reason", "")
        why_won = panel_decision.get("why_final_won", "")
        if conflict_reason:
            lines.append("  💬  Why panelists disagreed:")
            for l in _wrap_text(conflict_reason, 46, "       "):
                lines.append(l)
        if why_won:
            lines.append("  🏆  Why final diagnosis won:")
            for l in _wrap_text(why_won, 46, "       "):
                lines.append(l)
        lines.append("")

        # Alternates considered
        alternates = panel_decision.get("alternate_considered") or []
        if alternates:
            lines.append(f"  🔁  Alternates considered: {', '.join(alternates[:3])}")

        # Cannot-miss diagnoses
        cannot_miss = panel_decision.get("cannot_miss") or []
        if cannot_miss:
            lines.append(f"  ❗  Cannot-miss ruled out: {', '.join(cannot_miss[:2])}")

        # Resolving test
        resolving_test = panel_decision.get("resolving_test", "")
        if resolving_test:
            lines.append(f"  🔬  Key resolving test: {resolving_test}")

        # Uncertainty flag
        if uncertainty:
            lines.append("")
            lines.append("  ⚠️   Uncertainty remains — specialist referral recommended.")

        lines.append("")

    # ════ Risk Flags ════
    if risks:
        lines.append(SEP)
        lines.append("  ⚠️   RISK FLAGS")
        lines.append(SEP)
        for r in risks:
            lines.append(f"  ⚠  {r}")
        lines.append("")

    # ════ Recommended Tests ════
    if tests:
        lines.append(SEP)
        lines.append("  🔬  RECOMMENDED TESTS")
        lines.append(SEP)
        for t in tests:
            if isinstance(t, dict):
                lines.append(f"  ▸  {t.get('test_name', '')}")
                if t.get("reason"):
                    lines.append(f"       {t['reason']}")
            else:
                lines.append(f"  ▸  {t}")
        lines.append("")

    # ════ Emergency Contacts ════
    if emergency:
        contacts = response.get("emergency_contacts") or []
        if contacts:
            lines.append(SEP)
            lines.append("  📞  EMERGENCY CONTACTS")
            lines.append(SEP)
            for c in contacts:
                if isinstance(c, dict):
                    label = c.get("label", "")
                    num   = c.get("number", "")
                    if label and num:
                        lines.append(f"  📞  {label} : {num}")
            lines.append("")

    # ════ Footer ════
    lines.append(DSEП)
    lines.append(f"  ⚕  This is an AI-assisted assessment.")
    lines.append(f"     Always consult a qualified medical professional.")
    lines.append(DSEП)
    lines.append("")

    pretty_text = "\n".join(lines)

    # ── Summary string ──
    top_diag_str = ""
    if diagnoses:
        top = diagnoses[0]
        top_diag_str = f"{top.get('disease')} ({float(top.get('confidence', 0)):.0%} confidence)"
    summary = f"Top diagnosis: {top_diag_str}. Symptoms: {', '.join(symptoms)}."
    if risks:
        summary += f" Risks: {', '.join(risks)}."

    return {
        "emergency":        emergency,
        "patient":          {"symptoms": symptoms, "age": age, "gender": gender, "duration_days": duration},
        "risks":            risks,
        "recommended_tests": tests,
        "immediate_actions": remedy_steps[:5],
        "hospitals":        hospitals,
        "panel_decision":   panel_decision,
        "summary":          summary,
        "pretty_text":      pretty_text,
        "raw": {k: v for k, v in response.items() if k not in ("chat_history", "user_input", "_agent_trace")},
    }


# ── Hospital Detail Formatter ─────────────────────────────────────────────────

def format_hospital_details(
    hospital_name: str,
    disease: str,
    details: Dict[str, Any],
) -> str:
    """Render rich hospital + doctor details as attractive terminal output."""
    W    = 56
    SEP  = "  " + "─" * (W - 4)
    DSEП = "  " + "═" * (W - 4)
    lines: List[str] = []

    specialty = details.get("_specialty", "specialist")
    found_via_search = details.get("_search_used", False)

    # ── Header ──
    lines.append("")
    lines.append(f"{'🏥  Hospital Details  🏥':^{W}}")
    lines.append(DSEП)
    lines.append(f"  🏨  {details.get('hospital_name') or hospital_name}")
    if details.get("address"):
        lines.append(f"  📍  {details['address']}")
    lines.append(f"  🩺  Diagnosis  : {disease}")
    lines.append(f"  👨‍⚕️  Specialty  : {specialty.title()}")
    lines.append("")

    # ── Contact Info ──
    lines.append(SEP)
    lines.append("  📞  CONTACT INFORMATION")
    lines.append(SEP)

    phones = details.get("phone_numbers") or []
    if phones:
        for ph in phones:
            lines.append(f"  📞  {ph}")
    else:
        lines.append("  📞  Contact number not found — call hospital directly.")

    emergency_num = details.get("emergency_number")
    if emergency_num:
        lines.append(f"  🚨  Emergency : {emergency_num}")

    website = details.get("website")
    if website:
        lines.append(f"  🌐  Website   : {website}")

    lines.append("")

    # ── Appointment Info ──
    appt = details.get("appointment_info")
    booking_url = details.get("booking_url")
    if appt or booking_url:
        lines.append(SEP)
        lines.append("  📅  APPOINTMENT")
        lines.append(SEP)
        if appt:
            for l in _wrap_text(appt, 46, "      "):
                lines.append(l)
        if booking_url:
            lines.append(f"  🔗  Book online : {booking_url}")
        lines.append("")

    # ── Departments ──
    departments = details.get("departments") or []
    if departments:
        lines.append(SEP)
        lines.append("  🏛️   DEPARTMENTS")
        lines.append(SEP)
        lines.append(f"  ▸  {', '.join(departments)}")
        lines.append("")

    # ── Doctors ──
    doctors = details.get("doctors") or []
    if doctors:
        lines.append(SEP)
        lines.append(f"  👨‍⚕️  {specialty.upper()} DOCTORS")
        lines.append(SEP)
        for doc in doctors:
            name  = doc.get("name") or "Doctor"
            qual  = doc.get("qualifications") or ""
            avail = doc.get("availability") or ""
            cont  = doc.get("contact") or ""
            purl  = doc.get("profile_url") or ""

            lines.append(f"  👤  {name}")
            if qual:
                lines.append(f"       🎓  {qual}")
            if avail:
                lines.append(f"       🕐  {avail}")
            if cont:
                lines.append(f"       📞  {cont}")
            if purl:
                lines.append(f"       🔗  {purl}")
            lines.append("")
    else:
        lines.append(SEP)
        lines.append(f"  👨‍⚕️  {specialty.upper()} DOCTORS")
        lines.append(SEP)
        lines.append("  No specific doctor profiles found online.")
        lines.append(f"  Please contact {hospital_name} directly to book")
        lines.append(f"  with a {specialty}.")
        lines.append("")

    # ── Summary ──
    summary = details.get("summary")
    if summary:
        lines.append(SEP)
        lines.append("  📋  WHY THIS HOSPITAL")
        lines.append(SEP)
        for l in _wrap_text(summary, 46, "      "):
            lines.append(l)
        lines.append("")

    # ── Source note ──
    if not found_via_search:
        lines.append("  ℹ️   Details based on AI knowledge — verify directly with hospital.")
    else:
        lines.append("  ℹ️   Details sourced from online search — verify before visiting.")

    # ── Footer ──
    lines.append(DSEП)
    lines.append("  ⚕  Always confirm doctor availability before visiting.")
    lines.append(DSEП)
    lines.append("")

    return "\n".join(lines)
