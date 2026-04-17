from typing import TypedDict, Optional, Dict, Any, List


class AgentState(TypedDict):
    user_input: str
    chat_history: List[Dict[str, str]]
    session_memory: Optional[Dict[str, Any]]
    patient: Optional[Dict]
    location: Optional[Dict]
    hospitals: Optional[List[Dict[str, Any]]]
    need_location: Optional[bool]
    location_prompt: Optional[str]
    diagnosis: Optional[Dict]
    risks: Optional[Dict]
    tests: Optional[Dict]
    remedy: Optional[Dict]
    emergency_contacts: Optional[List[Dict[str, Any]]]
    is_emergency: bool
    is_medical_query: bool
    # ── Medical Panel fields ──────────────────────────────────────────────────
    panel_opinions: Optional[List[Dict[str, Any]]]    # raw output from 4 role agents
    panel_conflicts: Optional[Dict[str, Any]]          # conflict detector report
    panel_decision: Optional[Dict[str, Any]]           # adjudicator final output
    # ── Conversation routing fields ───────────────────────────────────────────
    question_type: Optional[str]                       # "followup" | "new_complaint"
    followup_answer: Optional[str]                     # plain-text answer for follow-up Qs