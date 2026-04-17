"""Question Classifier — routes each user turn to the right flow.

Acts as the new graph entry point (replaces domain_expert as the first node).

Question types:
  "followup"      → user is asking about their existing diagnosis/treatment
  "new_complaint" → user is describing new/different symptoms

The classifier is context-aware: it only considers a message a follow-up
when a prior diagnosis exists in session memory.
"""
import json
from typing import Dict, Any

from app.config import llm_call


def question_classifier(state: Dict[str, Any]) -> Dict[str, Any]:
    """Classify whether the user's message is a follow-up or a new complaint.

    Returns: {"question_type": "followup" | "new_complaint"}
    """
    user_input   = (state.get("user_input") or "").strip()
    chat_history = state.get("chat_history") or []
    session      = state.get("session_memory") or {}

    # If there's no prior diagnosis context at all, it must be a new complaint
    has_prior_diagnosis = bool(
        session.get("last_diagnosis")
        or session.get("last_diagnosis_full")
        or state.get("diagnosis")
    )
    if not has_prior_diagnosis or not chat_history:
        return {"question_type": "new_complaint"}

    # Pull the last known diagnosis name for context
    last_diag = session.get("last_diagnosis") or ""
    if isinstance(last_diag, dict):
        diags = last_diag.get("diagnoses", [])
        last_diag = diags[0].get("disease", "") if diags else ""

    # Build a short recent history string
    recent = chat_history[-4:]

    prompt = f"""You are a medical assistant query classifier. The patient has already received a diagnosis and is continuing their consultation.

Your job: decide if the patient's new message is asking a FOLLOW-UP question about their existing condition, or is describing BRAND NEW symptoms (a new complaint).

Previous diagnosis: {last_diag or "unknown"}
Recent conversation (last few messages):
{json.dumps(recent, indent=2)}

Patient's new message: "{user_input}"

FOLLOW-UP examples (question_type = "followup"):
- "what are the remedies for this?"
- "what medicines should I take?"
- "is this condition serious?"
- "what food should I avoid?"
- "how long will this take to heal?"
- "tell me more about treatment"
- "what are the precautions?"
- "can I exercise with this condition?"
- "what are the symptoms I should watch for?"
- "how can I manage the pain at home?"
- "should I be worried?"
- "what does this diagnosis mean?"

NEW COMPLAINT examples (question_type = "new_complaint"):
- "I now have joint pain and swelling"
- "my child is vomiting since morning"
- "I am having severe headache and fever"
- "I feel chest tightness now"

Return ONLY valid JSON:
{{"question_type": "followup"}}
OR
{{"question_type": "new_complaint"}}
"""

    try:
        response = llm_call(prompt)
        # Try to parse JSON
        try:
            data = json.loads(response)
        except Exception:
            start, end = response.find("{"), response.rfind("}") + 1
            data = json.loads(response[start:end]) if start != -1 and end > start else {}

        qt = data.get("question_type", "new_complaint")
        if qt not in ("followup", "new_complaint"):
            qt = "new_complaint"
        return {"question_type": qt}

    except Exception:
        return {"question_type": "new_complaint"}
