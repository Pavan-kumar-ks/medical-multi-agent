from app.schemas.patient import PatientData
from app.config import llm_call
import json

def _safe_parse_triage(response: str) -> bool:
    """Safely parse the boolean from the LLM response."""
    try:
        data = json.loads(response)
        if isinstance(data, dict) and "is_emergency" in data and isinstance(data["is_emergency"], bool):
            return data["is_emergency"]
    except json.JSONDecodeError:
        # Fallback for non-JSON responses
        if "true" in response.lower():
            return True
    return False

def triage_agent(patient: PatientData) -> dict:
    """
    An LLM-based agent to detect immediate emergencies from structured data
    by understanding the semantic meaning of the symptoms.
    """
    
    prompt = f"""
You are an experienced ER triage nurse. Your only task is to determine if the patient's situation is a critical emergency requiring immediate attention.
Base your decision on the semantic meaning of the symptoms.

Patient Data:
{json.dumps(patient.model_dump(), indent=2)}

Is this a critical emergency?
Answer with ONLY a valid JSON object in the following format:
{{
  "is_emergency": boolean
}}
"""

    response = llm_call(prompt)
    is_emergency = _safe_parse_triage(response)
    
    return {"is_emergency": is_emergency}
