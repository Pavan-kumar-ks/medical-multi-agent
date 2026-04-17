"""Safety Triage Lead — prioritises worst-case life-threatening conditions.

Independent role: sees patient data + evidence only, not other panelists.
Safety override rule: if this agent flags emergency, it always triggers emergency protocol.
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
        "emergency_override": False,
        "cannot_miss_diagnoses": [],
    }


def safety_triage_lead(state: Dict[str, Any]) -> Dict[str, Any]:
    """Role: Prioritises worst-case conditions and sets urgency.

    Safety rules enforced:
    1. Never suppress high-risk low-probability diagnoses without rationale.
    2. If ANY life-threatening condition is plausible, set emergency_override = true.
    3. Identify "cannot miss" diagnoses — conditions that must be ruled out first.
    """
    patient = PatientData(**state["patient"])
    chat_history = state.get("chat_history", [])[-4:]

    symptoms_text = " ".join([str(s) for s in patient.symptoms if s])
    context = retrieve_context(symptoms_text or str(patient.model_dump()))
    evidence_lines = [
        {"id": e.get("id") or e.get("source"), "text": (e.get("text") or str(e))[:300]}
        for e in context
    ]

    prompt = f"""You are the SAFETY TRIAGE LEAD on a medical expert panel.

Your role: Think worst-case first. Your priority is to identify any life-threatening diagnoses that MUST NOT be missed, even if their probability is low. You are responsible for patient safety above all else.

Safety principle: A low-probability high-severity diagnosis must be listed and flagged. Err on the side of caution.

Patient Data:
{json.dumps(patient.model_dump(), indent=2)}

Recent Conversation:
{json.dumps(chat_history, indent=2)}

Medical Evidence Available:
{json.dumps(evidence_lines, indent=2)}

Instructions:
- List up to 3 diagnoses — prioritise by SEVERITY, not just likelihood.
- Include at least one "cannot miss" diagnosis (even if unlikely).
- Set emergency_override = true if ANY listed diagnosis could be immediately life-threatening.
- List ALL red flags you see, even minor ones.
- Recommend the single most important test to rule out the worst-case diagnosis.
- Set urgency: "routine", "urgent", or "emergency".

Return ONLY valid JSON:
{{
  "role": "safety_triage_lead",
  "diagnoses": [
    {{
      "disease": "string",
      "confidence": 0.0,
      "reason": "why this must not be missed / severity justification",
      "evidence_refs": ["evidence_id"]
    }}
  ],
  "cannot_miss_diagnoses": ["string — conditions to rule out before anything else"],
  "red_flags": ["string"],
  "tests_needed": ["string — most critical safety-ruling test first"],
  "urgency": "routine|urgent|emergency",
  "emergency_override": false,
  "notes": "safety summary and recommended immediate actions if emergency"
}}
"""

    response = llm_call(prompt)
    result = _safe_parse(response, "safety_triage_lead")

    # Enforce: if urgency is emergency, always set emergency_override
    if result.get("urgency") == "emergency":
        result["emergency_override"] = True

    return result
