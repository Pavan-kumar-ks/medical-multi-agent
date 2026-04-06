import json
from typing import List

from app.config import llm_call
from app.schemas.patient import PatientData
from app.schemas.diagnosis import DiagnosisOutput, DiagnosisItem
from app.tools.retriever import retrieve_context


def _safe_parse_json(response: str):
    """
    Safely extract JSON from LLM response
    Handles cases where model adds extra text
    """
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # محاولة تنظيف الرد (common fix)
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                return json.loads(response[start:end])
            except:
                pass
    return None


def _normalize_confidence(value):
    """
    Ensure confidence is float between 0 and 1
    """
    try:
        val = float(value)
        return max(0.0, min(1.0, val))
    except:
        return 0.0


def diagnosis_agent(state: dict) -> DiagnosisOutput:
    """
    Diagnosis Agent with RAG (Retrieval-Augmented Generation) that is aware of chat history.
    """
    from app.schemas.patient import PatientData
    patient = PatientData(**state["patient"])
    chat_history = state.get("chat_history", [])

    # 🔍 Step 1: Retrieve medical context
    query = str(patient.model_dump())
    context: List[str] = retrieve_context(query)

    # 🧠 Step 2: Build prompt
    prompt = f"""
You are a clinical decision support assistant. Your role is to provide a diagnosis based on the patient's data and the conversation history.

Conversation History:
{json.dumps(chat_history, indent=2)}

Patient data:
{patient.model_dump()}

Relevant medical knowledge:
{context}

Instructions:
- Use the provided medical knowledge and the full conversation history to guide your reasoning.
- If the user is asking a follow-up question, use the history to inform your answer.
- Suggest top 3 possible diagnoses.
- Base reasoning on symptoms + retrieved knowledge + conversation context.
- Assign confidence between 0 and 1.
- Do NOT hallucinate diseases outside the context unless strongly justified.

Return ONLY valid JSON (no explanation):

{{
  "diagnoses": [
    {{
      "disease": "",
      "reason": "",
      "confidence": 0.0
    }}
  ]
}}
"""

    # 🤖 Step 3: LLM call
    response = llm_call(prompt)

    # 🧪 Step 4: Parse response safely
    parsed = _safe_parse_json(response)

    if not parsed or "diagnoses" not in parsed:
        return DiagnosisOutput(
            diagnoses=[
                DiagnosisItem(
                    disease="Unknown",
                    reason="Failed to parse model response",
                    confidence=0.0
                )
            ]
        )

    # 🧼 Step 5: Normalize output
    cleaned_diagnoses = []

    for item in parsed["diagnoses"]:
        cleaned_diagnoses.append(
            DiagnosisItem(
                disease=item.get("disease", "Unknown"),
                reason=item.get("reason", "No reasoning provided"),
                confidence=_normalize_confidence(item.get("confidence", 0.0))
            )
        )

    return DiagnosisOutput(diagnoses=cleaned_diagnoses)