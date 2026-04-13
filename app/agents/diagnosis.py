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

    # 🔍 Step 1: Retrieve medical context (structured evidence)
    # Use a short symptom-focused query for retrieval (better matches KB)
    try:
        symptoms_text = " ".join([str(s) for s in patient.symptoms if s])
    except Exception:
        symptoms_text = None
    query = symptoms_text or str(patient.model_dump())
    context: List[dict] = retrieve_context(query)

    # Format retrieved evidence for inclusion in the prompt (id + short text)
    evidence_lines = []
    for e in context:
        eid = e.get("id") or e.get("source")
        txt = (e.get("text") or str(e))
        evidence_lines.append({"id": eid, "text": txt[:300]})

    # 🧠 Step 2: Build prompt
    # If the orchestrator provided verifier feedback on a previous attempt,
    # include it so the LLM can address the verifier's concerns during retries.
    verifier_feedback = state.get("verifier_feedback")

    def _format_feedback(fb):
        try:
            if hasattr(fb, "model_dump"):
                return json.dumps(fb.model_dump(), indent=2)
            return json.dumps(fb, indent=2)
        except Exception:
            try:
                return str(fb)
            except Exception:
                return "<unserializable feedback>"

    feedback_section = ""
    if verifier_feedback:
        feedback_section = f"\nVerifier Feedback:\n{_format_feedback(verifier_feedback)}\n\n"

    # include a clear evidence section that the model can reference by id
    evidence_section = json.dumps(evidence_lines, indent=2, ensure_ascii=False)

    prompt = f"""
You are a clinical decision support assistant. Your role is to provide a diagnosis based on the patient's data and the conversation history.

Conversation History:
{json.dumps(chat_history, indent=2)}

Patient data:
{patient.model_dump()}

Relevant medical knowledge (list of evidence items with `id` and `text`):
{evidence_section}

{feedback_section}Instructions:
- Use the provided medical knowledge and the full conversation history to guide your reasoning.
- If the user is asking a follow-up question, use the history to inform your answer.
- Suggest top 3 possible diagnoses.
- Base reasoning on symptoms + retrieved knowledge + conversation context.
- Assign confidence between 0 and 1.
- Do NOT hallucinate diseases outside the context unless strongly justified.

Return ONLY valid JSON (no explanation). Each diagnosis must include `evidence_refs`,
which is a list of evidence `id` strings from the provided Relevant medical knowledge above.

Example:
- Return a single JSON object with a top-level key named "diagnoses".
- "diagnoses" must be a list of diagnosis objects.
- Each diagnosis object must include the keys: "disease" (string), "reason" (string),
  "confidence" (number between 0 and 1), and "evidence_refs" (list of evidence id strings).
Ensure the model outputs only valid JSON that exactly matches this schema (no surrounding explanation).
"""

    # 🤖 Step 3: LLM call
    response = llm_call(prompt)

    # 🧪 Step 4: Parse response safely
    parsed = _safe_parse_json(response)

    if not parsed or "diagnoses" not in parsed:
        # Rule-based fallback when LLM parsing fails: derive simple diagnoses
        text_symptoms = " ".join([str(s) for s in getattr(patient, "symptoms", [])]).lower()
        rules = [
            (["fever", "cold", "cough"], "Acute viral upper respiratory infection", "Symptoms consistent with an acute viral URI", 0.6),
            (["headache"], "Tension Headache", "Common primary headache related to stress/tension", 0.5),
            (["chest", "chest pain", "shortness"], "Cardiac ischemia (needs evaluation)", "Chest pain with dyspnea requires urgent evaluation", 0.1),
        ]
        diagnoses = []
        for keys, disease, reason, conf in rules:
            if any(k in text_symptoms for k in keys):
                # attach any retrieved evidence ids when available
                evidence_ids = [e.get("id") for e in context][:3]
                diagnoses.append(DiagnosisItem(disease=disease, reason=reason, confidence=conf, evidence_refs=evidence_ids))
        if diagnoses:
            return DiagnosisOutput(diagnoses=diagnoses)
        return DiagnosisOutput(
            diagnoses=[
                DiagnosisItem(
                    disease="Unknown",
                    reason="Failed to parse model response",
                    confidence=0.0,
                    evidence_refs=[],
                )
            ]
        )

    # 🧼 Step 5: Normalize output
    cleaned_diagnoses = []

    # Prepare simple auto-linking helpers: use patient symptoms + diagnosis words
    symptom_tokens = []
    try:
        symptom_tokens = [tok.lower() for s in patient.symptoms for tok in str(s).split()]
    except Exception:
        symptom_tokens = []

    for item in parsed["diagnoses"]:
        evid = item.get("evidence_refs") or []
        # If model didn't provide evidence_refs, attempt keyword-based matching
        if not evid:
            matches = []
            # search evidence_lines (constructed from `context`) for keywords
            for e in evidence_lines:
                text = (e.get("text") or "").lower()
                # match any symptom token or disease token
                disease_tokens = [tok.lower() for tok in str(item.get("disease", "")).split()]
                tokens = set(symptom_tokens + disease_tokens)
                if any(tok in text for tok in tokens if tok and len(tok) > 2):
                    matches.append(e.get("id"))
                if len(matches) >= 3:
                    break
            evid = matches

        cleaned_diagnoses.append(
            DiagnosisItem(
                disease=item.get("disease", "Unknown"),
                reason=item.get("reason", "No reasoning provided"),
                confidence=_normalize_confidence(item.get("confidence", 0.0)),
                evidence_refs=evid,
            )
        )

    return DiagnosisOutput(diagnoses=cleaned_diagnoses)