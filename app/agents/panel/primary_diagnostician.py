"""Primary Diagnostician — proposes the most likely differential diagnoses.

Independent role: sees patient data + evidence only, not other panelists.
"""
import json
from typing import Dict, Any

from app.config import llm_call
from app.tools.retriever import retrieve_context
from app.schemas.patient import PatientData


def _safe_parse(response: str, role: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, with fallback skeleton."""
    try:
        return json.loads(response)
    except Exception:
        start, end = response.find("{"), response.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(response[start:end])
            except Exception:
                pass
    return {
        "role": role,
        "diagnoses": [],
        "red_flags": [],
        "tests_needed": [],
        "urgency": "routine",
        "notes": "Parse error — no structured output returned.",
    }


def primary_diagnostician(state: Dict[str, Any]) -> Dict[str, Any]:
    """Role: Proposes top differential diagnoses based on symptoms + evidence.

    Does NOT see other panelists' opinions (avoids groupthink).
    Returns a structured PanelOpinion dict.
    """
    patient = PatientData(**state["patient"])
    chat_history = state.get("chat_history", [])[-4:]

    symptoms_text = " ".join([str(s) for s in patient.symptoms if s])
    context = retrieve_context(symptoms_text or str(patient.model_dump()))
    evidence_lines = [
        {"id": e.get("id") or e.get("source"), "text": (e.get("text") or str(e))[:300]}
        for e in context
    ]

    prompt = f"""You are the PRIMARY DIAGNOSTICIAN on a medical expert panel.

Your role: Propose the most clinically likely diagnoses based on the patient's symptoms, history, and retrieved medical evidence.

Patient Data:
{json.dumps(patient.model_dump(), indent=2)}

Recent Conversation:
{json.dumps(chat_history, indent=2)}

Medical Evidence Available:
{json.dumps(evidence_lines, indent=2)}

Instructions:
- Propose exactly 3 differential diagnoses, ranked by likelihood.
- Assign a confidence score (0.0 to 1.0) for each.
- Reference evidence item IDs that support each diagnosis.
- List clinical red flags you observe.
- Recommend the minimum tests needed to confirm your top diagnosis.
- Set urgency: "routine" (days), "urgent" (hours), or "emergency" (minutes).

Return ONLY valid JSON — no explanations outside the JSON:
{{
  "role": "primary_diagnostician",
  "diagnoses": [
    {{
      "disease": "string",
      "confidence": 0.0,
      "reason": "clinical reasoning string",
      "evidence_refs": ["evidence_id"]
    }}
  ],
  "red_flags": ["string"],
  "tests_needed": ["string"],
  "urgency": "routine|urgent|emergency",
  "notes": "any additional clinical note"
}}
"""

    response = llm_call(prompt)
    return _safe_parse(response, "primary_diagnostician")
