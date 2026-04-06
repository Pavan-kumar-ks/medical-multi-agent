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
    """
    
    prompt = f"""
You are a domain expert responsible for ensuring that this system only responds to medical queries.
Your task is to classify the user's input as either a medical query or a non-medical query.

User Input: "{user_input}"

Is this a medical query?
Answer with ONLY a valid JSON object in the following format:
{{
  "is_medical_query": boolean
}}
"""

    response = llm_call(prompt)
    is_medical = _safe_parse_domain(response)
    
    return {"is_medical_query": is_medical}
