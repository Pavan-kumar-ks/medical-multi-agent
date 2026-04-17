def route_after_diagnosis(state):
    """Route after hospital node based on panel-adjudicated confidence.

    Priority:
    1. If panel decision exists, use its final_score of top diagnosis.
    2. If panel flagged uncertainty, always route to tests.
    3. Fallback to original diagnosis confidence.
    """
    # ── Panel decision takes precedence ──
    panel = state.get("panel_decision") or {}
    if panel:
        # If adjudicator flagged uncertainty → more tests needed
        if panel.get("uncertainty_flag"):
            return "tests"

        final_diagnoses = panel.get("final_diagnoses") or []
        if final_diagnoses:
            # Use final_score (0–1) if available, else confidence
            top = final_diagnoses[0]
            top_score = top.get("final_score") or top.get("confidence", 0)
            return "risk" if float(top_score) >= 0.55 else "tests"

    # ── Fallback: use raw diagnosis confidence ──
    diagnosis = state.get("diagnosis") or {}
    if not diagnosis:
        return "tests"

    diagnoses = diagnosis.get("diagnoses", [])
    if diagnoses:
        top_conf = float(diagnoses[0].get("confidence", 0))
        return "risk" if top_conf >= 0.6 else "tests"

    return "tests"