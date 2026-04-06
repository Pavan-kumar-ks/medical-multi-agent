from app.config import llm_call
from app.schemas.patient import PatientData
import json

def emergency_remedy_agent(patient: PatientData) -> dict:
    """
    An LLM-based agent that provides immediate first-aid advice for emergencies.
    """
    
    prompt = f"""
You are an emergency first-aid advisor. Your task is to provide immediate, clear, and actionable steps for a medical emergency based on the patient's symptoms.
Your advice should be for a layperson and focus on what can be done while waiting for professional medical help.

Patient Data:
{json.dumps(patient.model_dump(), indent=2)}

Based on these symptoms, provide a list of immediate first-aid steps.
The advice should be concise and easy to follow.

Return ONLY a valid JSON object in the following format:
{{
  "remedy_steps": []
}}
"""

    response = llm_call(prompt)
    
    try:
        data = json.loads(response)
        return data
    except json.JSONDecodeError:
        return {"remedy_steps": ["Call emergency services immediately."]}
