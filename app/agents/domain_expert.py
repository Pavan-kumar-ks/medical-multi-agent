from app.config import llm_call
import json

def _safe_parse_domain(response: str) -> bool:
    """Safely parse the boolean from the LLM response."""
    try:
        data = json.loads(response)
        if isinstance(data, dict) and "is_medical_query" in data and isinstance(data["is_medical_query"], bool):
            return data["is_medical_query"]
    except json.JSONDecodeError:
        if "true" in response.lower():
            return True
    return False

def domain_expert_agent(user_input: str) -> dict:
    """
    An LLM-based agent to classify if the user input is a medical query.
    Handles mixed location+symptom inputs correctly.
    """

    prompt = f"""
You are a domain expert for a medical assistance system. Your job is to decide whether the user's message describes a medical concern, symptom, or health-related query.

IMPORTANT RULES:
- If the user describes ANY symptom, pain, discomfort, illness, or health concern — classify it as medical, even if they also mention a location, landmark, or hospital name.
- Mentions of nearby places (e.g. "near AIIMS", "I'm in Bangalore", "near City Hospital") alongside symptoms are STILL medical queries.
- Vague but health-related descriptions ("not feeling well", "feeling sick", "something is wrong") are medical queries.
- Only classify as NON-medical if the input has absolutely NO health or symptom information (e.g. weather, cooking, sports).

Examples:
  INPUT: "I have chest pain and shortness of breath"        → {{"is_medical_query": true}}
  INPUT: "I am near AIIMS and feeling very dizzy"           → {{"is_medical_query": true}}
  INPUT: "sudden breathlessness, near City Hospital"        → {{"is_medical_query": true}}
  INPUT: "fever, body ache and headache since 2 days"       → {{"is_medical_query": true}}
  INPUT: "I have been having chest tightness near my home"  → {{"is_medical_query": true}}
  INPUT: "what is the capital of France"                    → {{"is_medical_query": false}}
  INPUT: "how do I cook biryani"                            → {{"is_medical_query": false}}
  INPUT: "latest cricket score"                             → {{"is_medical_query": false}}

User Input: "{user_input}"

Answer with ONLY a valid JSON object:
{{
  "is_medical_query": boolean
}}
"""

    response = llm_call(prompt)
    is_medical = _safe_parse_domain(response)
    
    return {"is_medical_query": is_medical}
