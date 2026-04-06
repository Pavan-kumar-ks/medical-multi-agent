import json

from app.config import llm_call
from app.schemas.patient import PatientData


def _safe_parse(response: str):
    try:
        return json.loads(response)
    except:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                return json.loads(response[start:end])
            except:
                pass
    return None


def test_recommender_agent(patient: PatientData):
    prompt = f"""
You are a primary care doctor.

Patient data:
{patient.model_dump()}

Instructions:
- Suggest ONLY 3 to 5 MOST relevant tests
- Avoid unnecessary or excessive tests
- Prioritize cost-effective and essential diagnostics
- Do NOT include conditional tests (like "if symptoms present")

Return ONLY JSON:
{{
  "tests": [
    {{
      "test_name": "",
      "reason": ""
    }}
  ]
}}
"""

    response = llm_call(prompt)
    parsed = _safe_parse(response)

    if not parsed or "tests" not in parsed:
        return {
            "tests": [
                {
                    "test_name": "General physician evaluation",
                    "reason": "Unable to determine tests"
                }
            ]
        }

    return parsed