from langgraph.graph import StateGraph, END
from app.orchestrator.state import AgentState
from app.agents.domain_expert import domain_expert_agent
from app.agents.intake import intake_agent
from app.agents.triage import triage_agent
from app.agents.diagnosis import diagnosis_agent
from app.agents.verifier import verifier_agent
from app.agents.risk_analyzer import risk_analyzer_agent
from app.agents.test_recommender import test_recommender_agent
from app.agents.emergency_remedy_agent import emergency_remedy_agent
from app.agents.location_intake import location_intake_agent
from app.agents.hospital_finder import hospital_finder_agent
from app.orchestrator.router import route_after_diagnosis
from app.orchestrator.agent_runner import call_agent
# Panel agents
from app.agents.panel.primary_diagnostician import primary_diagnostician
from app.agents.panel.skeptical_reviewer import skeptical_reviewer
from app.agents.panel.evidence_auditor import evidence_auditor
from app.agents.panel.safety_triage_lead import safety_triage_lead
from app.agents.panel.conflict_detector import conflict_detector
from app.agents.panel.adjudicator import adjudicator
# Conversation routing agents
from app.agents.question_classifier import question_classifier
from app.agents.followup_responder import followup_responder

def build_graph():
    graph = StateGraph(AgentState)

    # Nodes

    def question_classifier_node(state):
        """New graph entry point — decides followup vs new complaint."""
        res = call_agent(
            question_classifier, args=(state,), retries=1,
            fallback={"question_type": "new_complaint"},
        )
        payload = res.get("result") or {}
        qt = payload.get("question_type", "new_complaint")
        return {"question_type": qt}

    def followup_responder_node(state):
        """Answer a follow-up question using prior diagnosis context."""
        res = call_agent(
            followup_responder, args=(state,), retries=1,
            fallback={"followup_answer": "I'm sorry, I couldn't generate a specific answer. Please consult your doctor."},
        )
        payload = res.get("result") or {}
        return {"followup_answer": payload.get("followup_answer", "")}

    def domain_expert_node(state):
        # Wrap domain expert in safe runner; fallback to non-medical
        res = call_agent(domain_expert_agent, args=(state["user_input"],), retries=1, fallback={"is_medical_query": False})
        payload = res.get("result") if res.get("ok") else res.get("result")
        is_medical = False
        try:
            if isinstance(payload, dict) and "is_medical_query" in payload:
                is_medical = bool(payload["is_medical_query"])
        except Exception:
            is_medical = False
        return {"is_medical": is_medical, "is_medical_query": is_medical}

    def intake_node(state):
        # Run intake with fallback that preserves the raw user input as a symptom
        fallback_patient = None
        try:
            from app.schemas.patient import PatientData
            fallback_patient = PatientData(symptoms=[state.get("user_input")])
        except Exception:
            fallback_patient = {"symptoms": [state.get("user_input")]}

        res = call_agent(intake_agent, args=(state,), retries=1, fallback=fallback_patient)
        patient_obj = res.get("result") if res.get("ok") else res.get("result")
        # If Pydantic object, normalize to dict
        try:
            patient_dict = patient_obj.model_dump() if hasattr(patient_obj, "model_dump") else dict(patient_obj)
        except Exception:
            patient_dict = {"symptoms": [state.get("user_input")]}
        return {"patient": patient_dict}

    def location_node(state):
        # Ask for location before diagnosis; if missing, prompt user
        res = call_agent(location_intake_agent, args=(state,), retries=0, fallback={"need_location": True, "location_prompt": "Please provide your location."})
        payload = res.get("result") if res.get("ok") else res.get("result")

        # Update session memory if provided
        session_memory = payload.get("session_memory") if isinstance(payload, dict) else None
        out = {}
        if isinstance(payload, dict):
            out.update({
                "location": payload.get("location"),
                "need_location": payload.get("need_location"),
                "location_prompt": payload.get("location_prompt"),
            })
        if session_memory is not None:
            out["session_memory"] = session_memory
        return out

    def triage_node(state):
        from app.schemas.patient import PatientData
        patient = PatientData(**state["patient"])
        # Wrap triage with fallback 'not emergency'
        res = call_agent(triage_agent, args=(patient,), retries=1, fallback={"is_emergency": False})
        payload = res.get("result") if res.get("ok") else res.get("result")
        is_emergency = False
        try:
            if isinstance(payload, dict) and "is_emergency" in payload:
                is_emergency = bool(payload["is_emergency"])
        except Exception:
            is_emergency = False
        return {"is_emergency": is_emergency}

    def diagnosis_node(state):
        # Diagnosis may fail; fallback to Unknown diagnosis
        # initial diagnosis attempt (agent_runner will itself retry once)
        res = call_agent(diagnosis_agent, args=(state,), retries=1, fallback=None)
        diag_obj = res.get("result") if res.get("ok") else res.get("result")

        try:
            diagnosis_dict = diag_obj.model_dump() if hasattr(diag_obj, "model_dump") else (dict(diag_obj) if diag_obj is not None else {"diagnoses": [{"disease": "Unknown", "reason": "Diagnosis failed", "confidence": 0.0, "evidence_refs": []}]})
        except Exception:
            diagnosis_dict = {"diagnoses": [{"disease": "Unknown", "reason": "Diagnosis failed", "confidence": 0.0, "evidence_refs": []}]}

        # Run verifier and allow the verifier to trigger additional diagnosis attempts
        verifier_attempts = 0
        max_verifier_retries = 2

        # helper to interpret verifier result (pydantic model or dict)
        def _is_verified(vres):
            if vres is None:
                return False
            try:
                ok = getattr(vres, "ok", None)
                if ok is None:
                    ok = vres.get("ok", False)
                return bool(ok)
            except Exception:
                return False

        # Prepare a mutable state copy including diagnosis for verification
        state_for_verifier = dict(state)
        state_for_verifier["diagnosis"] = diagnosis_dict

        vres = call_agent(verifier_agent, args=(state_for_verifier,), retries=0, fallback={"ok": False, "issues": ["verifier failed"], "per_item": []})
        verifier_result = vres.get("result") if vres.get("ok") else vres.get("result")

        # If not verified, attempt to re-run diagnosis up to max_verifier_retries
        while not _is_verified(verifier_result) and verifier_attempts < max_verifier_retries:
            verifier_attempts += 1
            # Re-run diagnosis with one extra attempt. Provide verifier feedback
            # so the diagnosis agent can address the verifier's issues.
            state_with_feedback = dict(state)
            # pass verifier_result as-is if it's serializable, otherwise try model_dump
            try:
                if hasattr(verifier_result, "model_dump"):
                    state_with_feedback["verifier_feedback"] = verifier_result.model_dump()
                else:
                    state_with_feedback["verifier_feedback"] = verifier_result
            except Exception:
                state_with_feedback["verifier_feedback"] = str(verifier_result)

            r = call_agent(diagnosis_agent, args=(state_with_feedback,), retries=1, fallback=None)
            dobj = r.get("result") if r.get("ok") else r.get("result")
            try:
                diagnosis_dict = dobj.model_dump() if hasattr(dobj, "model_dump") else (dict(dobj) if dobj is not None else {"diagnoses": [{"disease": "Unknown", "reason": "Diagnosis failed", "confidence": 0.0, "evidence_refs": []}]})
            except Exception:
                diagnosis_dict = {"diagnoses": [{"disease": "Unknown", "reason": "Diagnosis failed", "confidence": 0.0, "evidence_refs": []}]}

            state_for_verifier["diagnosis"] = diagnosis_dict
            vres = call_agent(verifier_agent, args=(state_for_verifier,), retries=0, fallback={"ok": False, "issues": ["verifier failed"], "per_item": []})
            verifier_result = vres.get("result") if vres.get("ok") else vres.get("result")

        # If still not verified, replace diagnosis with safe fallback
        if not _is_verified(verifier_result):
            diagnosis_dict = {"diagnoses": [{"disease": "Unknown", "reason": "Failed verification", "confidence": 0.0, "evidence_refs": []}]}

        return {"diagnosis": diagnosis_dict}

    def risk_node(state):
        from app.schemas.patient import PatientData
        patient = PatientData(**state["patient"])
        # Risk analyzer wrapped with fallback empty risks
        res = call_agent(risk_analyzer_agent, args=(patient,), retries=1, fallback={"risks": []})
        payload = res.get("result") if res.get("ok") else res.get("result")
        try:
            risks = payload if isinstance(payload, list) else payload.get("risks", payload)
        except Exception:
            risks = []
        return {"risks": risks}

    def test_node(state):
        from app.schemas.patient import PatientData
        patient = PatientData(**state["patient"])
        # Tests recommender with fallback empty list
        res = call_agent(test_recommender_agent, args=(patient,), retries=1, fallback=[])
        payload = res.get("result") if res.get("ok") else res.get("result")
        try:
            tests = payload
        except Exception:
            tests = []
        return {"tests": tests}

    def emergency_node(state):
        print("--- EMERGENCY DETECTED ---")
        print("Generating immediate first-aid advice...")
        from app.schemas.patient import PatientData
        patient = PatientData(**state["patient"])
        # Emergency remedy should be best-effort; fallback to a generic instruction
        res = call_agent(emergency_remedy_agent, args=(patient,), retries=1, fallback={"remedy_steps": ["Call emergency services immediately."]})
        remedy = res.get("result") if res.get("ok") else res.get("result")
        # Also fetch dynamic emergency contact numbers from LLM
        try:
            from app.tools.emergency_contacts import fetch_emergency_contacts
            contacts = fetch_emergency_contacts({"patient": state.get("patient", {})})
        except Exception:
            contacts = []

        return {"remedy": remedy, "emergency_contacts": contacts}

    def panel_node(state):
        """Run all 4 panel role agents independently, then detect conflicts and adjudicate."""
        print("🔬 Running medical panel review...")

        # ── Step 1: Independent opinions (no cross-talk) ──
        fallback_opinion = {
            "role": "unknown", "diagnoses": [], "red_flags": [],
            "tests_needed": [], "urgency": "routine", "notes": "Agent failed.",
        }
        res_primary  = call_agent(primary_diagnostician,  args=(state,), retries=1, fallback=fallback_opinion)
        res_skeptic  = call_agent(skeptical_reviewer,      args=(state,), retries=1, fallback={**fallback_opinion, "role": "skeptical_reviewer"})
        res_auditor  = call_agent(evidence_auditor,        args=(state,), retries=1, fallback={**fallback_opinion, "role": "evidence_auditor"})
        res_safety   = call_agent(safety_triage_lead,      args=(state,), retries=1, fallback={**fallback_opinion, "role": "safety_triage_lead", "emergency_override": False, "cannot_miss_diagnoses": []})

        opinions = [
            res_primary.get("result") or fallback_opinion,
            res_skeptic.get("result") or {**fallback_opinion, "role": "skeptical_reviewer"},
            res_auditor.get("result") or {**fallback_opinion, "role": "evidence_auditor"},
            res_safety.get("result")  or {**fallback_opinion, "role": "safety_triage_lead"},
        ]

        # ── Step 2: Conflict detection ──
        print("⚖️  Detecting panel conflicts...")
        try:
            conflict_report = conflict_detector(opinions)
        except Exception as e:
            conflict_report = {
                "conflicts": [], "conflict_count": 0,
                "emergency_flagged": False, "all_urgencies": ["routine"],
                "all_top_diseases": [], "consensus_diseases": [],
            }

        # ── Step 3: Adjudication ──
        print("🏛️  Adjudicating panel decision...")
        try:
            panel_result = adjudicator(opinions, conflict_report)
        except Exception:
            panel_result = {
                "final_diagnoses": state.get("diagnosis", {}).get("diagnoses", []),
                "emergency_triggered": False,
                "resolved_urgency": "routine",
                "panel_summary": "Panel adjudication failed; original diagnosis used.",
                "conflict_reason": "", "why_final_won": "",
                "resolving_test": "", "alternate_considered": [],
                "uncertainty_flag": True, "conflict_count": 0,
                "consensus_diseases": [], "cannot_miss": [],
            }

        # ── Safety override: if panel triggers emergency, update is_emergency ──
        is_emergency = state.get("is_emergency", False)
        if panel_result.get("emergency_triggered"):
            is_emergency = True
            print("🚨 Panel triggered emergency override!")

        # ── Merge panel final_diagnoses back into the diagnosis field ──
        # This ensures downstream nodes (hospital, risk, tests) use panel-adjudicated diagnosis
        panel_diag = panel_result.get("final_diagnoses")
        updated_diagnosis = state.get("diagnosis") or {}
        if panel_diag:
            updated_diagnosis = {"diagnoses": panel_diag}

        return {
            "panel_opinions":  opinions,
            "panel_conflicts": conflict_report,
            "panel_decision":  panel_result,
            "diagnosis":       updated_diagnosis,
            "is_emergency":    is_emergency,
        }

    def hospital_node(state):
        res = call_agent(hospital_finder_agent, args=(state,), retries=0, fallback={"hospitals": []})
        payload = res.get("result") if res.get("ok") else res.get("result")
        try:
            hospitals = payload.get("hospitals", []) if isinstance(payload, dict) else payload
            meta = payload.get("hospital_search_meta", {}) if isinstance(payload, dict) else {}
        except Exception:
            hospitals = []
            meta = {}
        return {"hospitals": hospitals, "hospital_search_meta": meta}

    def await_location_node(state):
        # Placeholder node when waiting for user location input
        return {}

    def non_medical_node(state):
        print("This system is designed to answer medical queries only. Please provide a medical query.")
        return {}

    # Add nodes
    graph.add_node("question_classifier", question_classifier_node)   # ← new entry point
    graph.add_node("followup_responder",  followup_responder_node)     # ← follow-up handler
    graph.add_node("domain_expert", domain_expert_node)
    graph.add_node("intake", intake_node)
    graph.add_node("triage", triage_node)
    graph.add_node("location", location_node)
    graph.add_node("diagnosis", diagnosis_node)
    graph.add_node("panel", panel_node)
    graph.add_node("risk", risk_node)
    graph.add_node("tests", test_node)
    graph.add_node("emergency", emergency_node)
    graph.add_node("hospital", hospital_node)
    graph.add_node("await_location", await_location_node)
    graph.add_node("non_medical", non_medical_node)

    # ── Entry point: question classifier ──────────────────────────────────
    graph.set_entry_point("question_classifier")

    def route_after_classifier(state):
        if state.get("question_type") == "followup":
            return "followup_responder"
        return "domain_expert"

    graph.add_conditional_edges(
        "question_classifier",
        route_after_classifier,
        {
            "followup_responder": "followup_responder",
            "domain_expert":      "domain_expert",
        },
    )
    # Follow-up answers are complete — end here
    graph.add_edge("followup_responder", END)

    # ── Existing routing from domain expert ───────────────────────────────
    def route_after_domain_expert(state):
        if state.get("is_medical_query"):
            return "intake"
        return "non_medical"

    graph.add_conditional_edges(
        "domain_expert",
        route_after_domain_expert,
        {
            "intake": "intake",
            "non_medical": "non_medical",
        },
    )
    graph.add_edge("non_medical", END)

    graph.add_edge("intake", "location")

    def route_after_location(state):
        # If location missing, stop and ask user for it
        if state.get("need_location"):
            return "await_location"
        return "triage"

    graph.add_conditional_edges(
        "location",
        route_after_location,
        {
            "triage": "triage",
            "await_location": "await_location",
        }
    )

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
    
    # Emergency still gets diagnosis so hospitals can be ranked by diagnosis context
    graph.add_edge("emergency", "diagnosis")

    # Diagnosis → Panel → Hospital
    # Panel reviews, resolves conflicts, and may override emergency flag
    graph.add_edge("diagnosis", "panel")
    graph.add_edge("panel", "hospital")

    # Conditional routing after hospital (based on diagnosis confidence)
    graph.add_conditional_edges(
        "hospital",
        route_after_diagnosis,
        {
            "risk": "risk",
            "tests": "tests"
        }
    )

    graph.add_edge("risk", "tests")
    graph.add_edge("tests", END)

    return graph.compile()