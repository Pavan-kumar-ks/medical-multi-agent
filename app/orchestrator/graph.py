from langgraph.graph import StateGraph, END
from app.orchestrator.state import AgentState
from app.agents.domain_expert import domain_expert_agent
from app.agents.intake import intake_agent
from app.agents.triage import triage_agent
from app.agents.diagnosis import diagnosis_agent
from app.agents.risk_analyzer import risk_analyzer_agent
from app.agents.test_recommender import test_recommender_agent
from app.agents.emergency_remedy_agent import emergency_remedy_agent
from app.orchestrator.router import route_after_diagnosis

def build_graph():
    graph = StateGraph(AgentState)

    # Nodes
    def domain_expert_node(state):
        result = domain_expert_agent(state["user_input"])
        return {"is_medical_query": result["is_medical_query"]}

    def intake_node(state):
        patient = intake_agent(state)
        return {"patient": patient.model_dump()}

    def triage_node(state):
        from app.schemas.patient import PatientData
        patient = PatientData(**state["patient"])
        result = triage_agent(patient)
        return {"is_emergency": result["is_emergency"]}

    def diagnosis_node(state):
        result = diagnosis_agent(state)
        return {"diagnosis": result.model_dump()}

    def risk_node(state):
        from app.schemas.patient import PatientData
        patient = PatientData(**state["patient"])
        return {"risks": risk_analyzer_agent(patient)}

    def test_node(state):
        from app.schemas.patient import PatientData
        patient = PatientData(**state["patient"])
        return {"tests": test_recommender_agent(patient)}

    def emergency_node(state):
        print("--- EMERGENCY DETECTED ---")
        print("Generating immediate first-aid advice...")
        from app.schemas.patient import PatientData
        patient = PatientData(**state["patient"])
        remedy = emergency_remedy_agent(patient)
        return {"remedy": remedy}

    def non_medical_node(state):
        print("This system is designed to answer medical queries only. Please provide a medical query.")
        return {}

    # Add nodes
    graph.add_node("domain_expert", domain_expert_node)
    graph.add_node("intake", intake_node)
    graph.add_node("triage", triage_node)
    graph.add_node("diagnosis", diagnosis_node)
    graph.add_node("risk", risk_node)
    graph.add_node("tests", test_node)
    graph.add_node("emergency", emergency_node)
    graph.add_node("non_medical", non_medical_node)

    # Flow
    graph.set_entry_point("domain_expert")

    # Conditional routing after domain expert
    def route_after_domain_expert(state):
        if state.get("is_medical_query"):
            return "intake"
        return "non_medical"

    graph.add_conditional_edges(
        "domain_expert",
        route_after_domain_expert,
        {
            "intake": "intake",
            "non_medical": "non_medical"
        }
    )
    graph.add_edge("non_medical", END)

    graph.add_edge("intake", "triage")

    # Conditional routing after triage
    def route_after_triage(state):
        if state.get("is_emergency"):
            return "emergency"
        return "diagnosis"

    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "emergency": "emergency",
            "diagnosis": "diagnosis"
        }
    )
    
    graph.add_edge("emergency", "risk")

    # Conditional routing after diagnosis
    graph.add_conditional_edges(
        "diagnosis",
        route_after_diagnosis,
        {
            "risk": "risk",
            "tests": "tests"
        }
    )

    graph.add_edge("risk", "tests")
    graph.add_edge("tests", END)

    return graph.compile()