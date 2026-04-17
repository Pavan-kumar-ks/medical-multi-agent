"""Follow-up Responder — answers contextual questions about an existing diagnosis.

Activated when question_classifier returns "followup".
Does NOT re-run the full pipeline. Uses prior diagnosis + patient data
from session memory to give a targeted, specific answer.
"""
import json
from typing import Dict, Any, List

from app.config import llm_call


def _get_top_disease(diagnosis: Dict) -> str:
    try:
        diags = diagnosis.get("diagnoses", [])
        if diags:
            top = max(diags, key=lambda d: float(d.get("confidence", 0)))
            return top.get("disease", "")
    except Exception:
        pass
    return ""


def _safe_list(val) -> List:
    if isinstance(val, list):
        return val
    if isinstance(val, dict):
        # handle {"risks": [...]} or {"tests": [...]}
        for k in ("risks", "tests", "remedy_steps"):
            if k in val:
                return val[k]
    return []


def followup_responder(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a specific, helpful answer to a follow-up medical question.

    Pulls context from:
      - state["session_memory"]["last_diagnosis_full"]
      - state["session_memory"]["last_patient"]
      - state["session_memory"]["last_risks"]
      - state["session_memory"]["last_tests"]
      - state["chat_history"]
    """
    user_input   = (state.get("user_input") or "").strip()
    chat_history = state.get("chat_history") or []
    session      = state.get("session_memory") or {}

    # ── Gather context ────────────────────────────────────────────────────
    diagnosis = (
        state.get("diagnosis")
        or session.get("last_diagnosis_full")
        or session.get("last_diagnosis")
        or {}
    )
    patient = (
        state.get("patient")
        or session.get("last_patient")
        or {}
    )
    panel_decision = (
        state.get("panel_decision")
        or session.get("last_panel_decision")
        or {}
    )
    risks = _safe_list(state.get("risks") or session.get("last_risks") or [])
    tests = _safe_list(state.get("tests") or session.get("last_tests") or [])

    top_disease  = _get_top_disease(diagnosis)
    all_diagnoses = diagnosis.get("diagnoses", [])
    symptoms     = patient.get("symptoms", [])
    age          = patient.get("age")
    gender       = patient.get("gender")

    # Panel alternates (for richer context)
    alternates   = panel_decision.get("alternate_considered", [])

    # Recent conversation (trim to last 6 turns)
    recent_history = chat_history[-6:]

    # ── Build prompt ──────────────────────────────────────────────────────
    prompt = f"""You are a knowledgeable, empathetic medical assistant. The patient has already been evaluated and you now need to answer their follow-up question.

=== Patient Profile ===
Age: {age or "not provided"}
Gender: {gender or "not provided"}
Reported Symptoms: {', '.join(symptoms) if symptoms else "not recorded"}

=== Diagnosis Summary ===
Primary Diagnosis: {top_disease or "not determined"}
All Diagnoses Considered:
{json.dumps(all_diagnoses, indent=2)}
Alternate Conditions Considered: {', '.join(alternates) if alternates else "none"}
Risk Flags: {', '.join(risks) if risks else "none"}
Recommended Tests: {json.dumps(tests, indent=2) if tests else "none"}

=== Recent Conversation ===
{json.dumps(recent_history, indent=2)}

=== Patient's Follow-up Question ===
"{user_input}"

=== Instructions ===
Answer the patient's question SPECIFICALLY for their diagnosis ({top_disease}).
- If asking about remedies/treatment: give concrete home remedies + medical treatments for {top_disease}
- If asking about medications: list commonly prescribed medications (add disclaimer to get prescription)
- If asking about diet/food: give specific dietary advice for {top_disease}
- If asking about precautions/lifestyle: give actionable precautions specific to {top_disease}
- If asking about prognosis: explain recovery timeline and what to expect
- If asking about warning signs: list specific red flags that need immediate attention
- If asking "what does this mean": explain the diagnosis in plain, patient-friendly language

Be clear, specific, and compassionate. Format with numbered steps or bullets where helpful.
Do NOT restart the diagnosis process. Do NOT repeat the full assessment.
End with a one-line reminder to consult a doctor for personalised advice.
"""

    response = llm_call(prompt)
    return {"followup_answer": response.strip()}
