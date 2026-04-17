"""Skeptical Reviewer — challenges assumptions and hunts for missed diagnoses.

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


def skeptical_reviewer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Role: Challenges weak assumptions and surfaces under-considered diagnoses.

    Actively looks for:
    - Diagnoses that are easy to miss but serious.
    - Symptom patterns that could indicate something other than the obvious.
    - Gaps or overconfidence in the evidence base.
    """
    patient = PatientData(**state["patient"])
    chat_history = state.get("chat_history", [])[-4:]

    symptoms_text = " ".join([str(s) for s in patient.symptoms if s])
    context = retrieve_context(symptoms_text or str(patient.model_dump()))
    evidence_lines = [
        {"id": e.get("id") or e.get("source"), "text": (e.get("text") or str(e))[:300]}
        for e in context
    ]

    prompt = f"""You are the SKEPTICAL REVIEWER on a medical expert panel.

Your role: Challenge the obvious diagnosis. Actively look for what might be missed, misattributed, or underweighted. Your job is NOT to agree — it is to surface alternative possibilities and flag weak reasoning.

Patient Data:
{json.dumps(patient.model_dump(), indent=2)}

Recent Conversation:
{json.dumps(chat_history, indent=2)}

Medical Evidence Available:
{json.dumps(evidence_lines, indent=2)}

Instructions:
- Propose 3 diagnoses — prioritise ones that are LESS OBVIOUS but clinically plausible.
- Flag symptoms that could point to a more serious or different condition.
- Identify what symptoms or evidence are MISSING that would rule in/out serious diagnoses.
- Assign confidence (0.0 to 1.0) — be conservative if evidence is thin.
- List the minimum tests that would most effectively rule out the scariest possibility.
- Set urgency: "routine", "urgent", or "emergency".

Return ONLY valid JSON:
{{
  "role": "skeptical_reviewer",
  "diagnoses": [
    {{
      "disease": "string",
      "confidence": 0.0,
      "reason": "why this might be missed or underweighted",
      "evidence_refs": ["evidence_id"]
    }}
  ],
  "red_flags": ["string — things that concern you or are not explained"],
  "tests_needed": ["string — tests that would most change your assessment"],
  "urgency": "routine|urgent|emergency",
  "notes": "summary of your main concern or challenge"
}}
"""

    response = llm_call(prompt)
    return _safe_parse(response, "skeptical_reviewer")
