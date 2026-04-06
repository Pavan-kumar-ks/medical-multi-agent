from typing import TypedDict, Optional, Dict, Any, List


class AgentState(TypedDict):
    user_input: str
    chat_history: List[Dict[str, str]]
    patient: Optional[Dict]
    diagnosis: Optional[Dict]
    risks: Optional[Dict]
    tests: Optional[Dict]
    remedy: Optional[Dict]
    is_emergency: bool
    is_medical_query: bool