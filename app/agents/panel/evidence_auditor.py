"""Evidence Auditor — checks whether each diagnosis claim is supported by retrieved evidence.

Independent role: sees patient data + evidence only, not other panelists.
"""
import json
from typing import Dict, Any

from app.config import llm_call
from app.tools.retriever import retrieve_context
from app.schemas.patient import PatientData


def _safe_parse(response: str, role: str) -> Dict[str, Any]:
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


def evidence_auditor(state: Dict[str, Any]) -> Dict[str, Any]:
    """Role: Evaluates evidence quality for each possible diagnosis.

    Scores each diagnosis by:
    - How well the retrieved knowledge base supports it.
    - Whether symptom patterns match documented clinical criteria.
    - Where evidence is absent, ambiguous, or contradictory.
    """
    patient = PatientData(**state["patient"])
    chat_history = state.get("chat_history", [])[-4:]

    symptoms_text = " ".join([str(s) for s in patient.symptoms if s])
    context = retrieve_context(symptoms_text or str(patient.model_dump()))
    evidence_lines = [
        {"id": e.get("id") or e.get("source"), "text": (e.get("text") or str(e))[:300]}
        for e in context
    ]

    prompt = f"""You are the EVIDENCE AUDITOR on a medical expert panel.

Your role: Assess which diagnoses are best supported by the available medical evidence. You care about evidence quality, not clinical intuition. Flag any diagnosis that lacks supporting evidence.

Patient Data:
{json.dumps(patient.model_dump(), indent=2)}

Recent Conversation:
{json.dumps(chat_history, indent=2)}

Medical Evidence Available (these are the ONLY sources you can cite):
{json.dumps(evidence_lines, indent=2)}

Instructions:
- Propose 3 diagnoses that are BEST SUPPORTED by the evidence items above.
- For each, cite the specific evidence IDs that support it.
- If a diagnosis cannot be linked to at least one evidence item, give it lower confidence.
- Flag any symptoms or patterns NOT covered by the available evidence.
- Recommend tests that would add new evidence to resolve ambiguity.
- Set urgency: "routine", "urgent", or "emergency".

Return ONLY valid JSON:
{{
  "role": "evidence_auditor",
  "diagnoses": [
    {{
      "disease": "string",
      "confidence": 0.0,
      "reason": "which evidence items support this and how",
      "evidence_refs": ["evidence_id"],
      "evidence_strength": "strong|moderate|weak|none"
    }}
  ],
  "red_flags": ["string — symptom or claim not supported by any evidence"],
  "tests_needed": ["string — tests that would add new diagnostic evidence"],
  "urgency": "routine|urgent|emergency",
  "notes": "summary of evidence gaps or quality issues"
}}
"""

    response = llm_call(prompt)
    result = _safe_parse(response, "evidence_auditor")

    # Normalise: remove evidence_strength from nested diagnoses if present
    # (keep it for internal use; outer schema stays compatible)
    return result
