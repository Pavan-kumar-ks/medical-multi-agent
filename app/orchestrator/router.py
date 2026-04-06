def route_after_diagnosis(state):
    diagnosis = state.get("diagnosis", {})

    if not diagnosis:
        return "tests"

    diagnoses = diagnosis.get("diagnoses", [])

    # Check confidence
    if diagnoses:
        top_conf = diagnoses[0].get("confidence", 0)

        if top_conf < 0.6:
            return "tests"  # low confidence → more testing

    return "risk"