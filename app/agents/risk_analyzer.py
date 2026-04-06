from app.tools.rules_engine import apply_medical_rules
from app.schemas.patient import PatientData


def risk_analyzer_agent(patient: PatientData):
    risks = apply_medical_rules(patient)

    return {
        "risks": risks
    }