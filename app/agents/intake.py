import json
from app.config import llm_call
from app.schemas.patient import PatientData

def intake_agent(state: dict) -> PatientData:
    user_input = state["user_input"]
    chat_history = state.get("chat_history", [])

    prompt = f"""
You are a medical intake assistant. Your job is to extract structured information from the user's input, considering the entire conversation history.

Conversation History:
{json.dumps(chat_history, indent=2)}

Latest User Input: "{user_input}"

Based on the latest input and the conversation history, extract or update the patient's information.
If the user provides new information, update the corresponding fields. If they are asking a question, you can extract symptoms from their previous statements.

Return ONLY valid JSON in this format:
{{
    "symptoms": [],
    "duration_days": number or null,
    "severity": "mild/moderate/severe" or null,
    "age": number or null,
    "gender": "male/female/other" or null
}}
"""

    response = llm_call(prompt)

    try:
        data = json.loads(response)
        return PatientData(**data)
    except Exception:
        # fallback (very important)
        return PatientData(symptoms=[user_input])