import sys
import os
import re
import difflib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.orchestrator.graph import build_graph
from app.memory.vector_store import load_vector_store
import json
from app.tools.formatter import format_medical_response, format_hospital_details
from app.memory.session_memory import SessionMemory
from app.agents.location_intake import location_intake_agent
from app.agents.hospital_detail_agent import hospital_detail_agent
from app.orchestrator.agent_runner import reset_agent_trace, get_agent_trace


# ── Core graph runner ─────────────────────────────────────────────────────────

def run_agentic_system(user_input: str, chat_history: list, session_memory: SessionMemory | None = None):
    load_vector_store()
    graph = build_graph()
    reset_agent_trace()

    result = graph.invoke({
        "user_input":    user_input,
        "chat_history":  chat_history,
        "session_memory": session_memory.data if session_memory else {},
    })

    try:
        result["_agent_trace"] = get_agent_trace()
    except Exception:
        result["_agent_trace"] = []

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prompt_if_missing(session_memory: SessionMemory, key: str, prompt: str) -> bool:
    if not session_memory.get(key):
        print("\nAssistant:")
        print(prompt)
        print("\n" + "=" * 50 + "\n")
        return True
    return False


def _try_parse_age(text: str):
    try:
        age = int("".join([c for c in text if c.isdigit()]))
        return age if age > 0 else None
    except Exception:
        return None


_HOSPITAL_STOP_WORDS = {
    "hospital", "hospitals", "clinic", "clinics", "medical", "centre",
    "center", "health", "care", "healthcare", "the", "and", "of", "at",
    "limited", "ltd", "pvt", "private", "trust", "foundation",
}

# Words that suggest the user is trying to select a hospital
_HOSPITAL_INTENT_WORDS = {
    "hospital", "clinic", "medical", "health", "care", "centre", "center",
    "nursing", "institute", "apollo", "fortis", "manipal", "narayana",
    "columbia", "ramaiah", "ramaja", "bgS", "bgs", "m s ", "ms ",
    "dr ", "nelamangala", "bengaluru", "bangalore",
}


def _normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, remove stop words, collapse spaces."""
    tokens = re.sub(r"[^\w\s]", " ", name.lower()).split()
    core = [t for t in tokens if t not in _HOSPITAL_STOP_WORDS and len(t) > 1]
    return " ".join(core)


def _match_hospital(user_input: str, hospitals: list) -> dict | None:
    """Try to match the user's input to one of the shown hospitals.

    Accepts:
    1. A digit (1–5)
    2. Exact/substring name match
    3. Fuzzy token-normalised name match (difflib ratio ≥ 0.55)
    """
    text = user_input.strip()

    # ── 1. Number selection ──────────────────────────────────────────────
    if text.isdigit():
        idx = int(text) - 1
        if 0 <= idx < len(hospitals):
            return hospitals[idx]

    # ── 2. Exact substring match ─────────────────────────────────────────
    text_lower = text.lower()
    for h in hospitals:
        name = (h.get("name") or "").lower()
        if name and (name in text_lower or text_lower in name):
            return h

    # ── 3. Fuzzy normalised match ─────────────────────────────────────────
    user_norm = _normalize_name(text)
    if user_norm:
        best_score, best_match = 0.0, None
        for h in hospitals:
            hosp_norm = _normalize_name(h.get("name", ""))
            if not hosp_norm:
                continue
            ratio = difflib.SequenceMatcher(None, user_norm, hosp_norm).ratio()
            if ratio > best_score:
                best_score = ratio
                best_match = h
        if best_score >= 0.50:
            return best_match

    return None


def _best_fuzzy_hospital(user_input: str, hospitals: list) -> dict | None:
    """Return the closest-matching hospital even below the confident threshold."""
    user_norm = _normalize_name(user_input)
    if not user_norm or not hospitals:
        return None
    return max(
        hospitals,
        key=lambda h: difflib.SequenceMatcher(
            None, user_norm, _normalize_name(h.get("name", ""))
        ).ratio(),
        default=None,
    )


def _looks_like_hospital_query(text: str) -> bool:
    """Return True when the text likely refers to a hospital name or selection."""
    t = text.lower()
    # Single digit is clearly a selection attempt
    if text.strip().isdigit():
        return True
    return any(kw in t for kw in _HOSPITAL_INTENT_WORDS)


def _store_diagnosis_context(session_memory: SessionMemory, result: dict):
    """Persist diagnosis, patient, and panel info for follow-up questions."""
    if result.get("diagnosis"):
        session_memory.set("last_diagnosis_full", result["diagnosis"])
        # also keep a lightweight copy for the question_classifier check
        diags = result["diagnosis"].get("diagnoses", [])
        session_memory.set("last_diagnosis", result["diagnosis"])
        if diags:
            top = max(diags, key=lambda d: float(d.get("confidence", 0)))
            session_memory.set("last_top_disease", top.get("disease", ""))
    if result.get("patient"):
        session_memory.set("last_patient", result["patient"])
    if result.get("panel_decision"):
        session_memory.set("last_panel_decision", result["panel_decision"])
    if result.get("risks"):
        session_memory.set("last_risks", result["risks"])
    if result.get("tests"):
        session_memory.set("last_tests", result["tests"])


# ── Welcome banner ───────────────────────────────────────────────────────────

_WELCOME_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        🏥   Welcome to  M e d O r c h e s t r a t o r      ║
║                                                              ║
║     AI-Powered Multi-Agent Clinical Decision System          ║
║                                                              ║
║  ✦  4-Agent Medical Panel with conflict resolution          ║
║  ✦  Evidence-based diagnosis with confidence scoring        ║
║  ✦  Nearest hospital finder with doctor search              ║
║  ✦  Real-time web scraping for doctor profiles              ║
║                                                              ║
║  Type  'quit'  or  'exit'  at any time to end the session.  ║
╚══════════════════════════════════════════════════════════════╝
"""


# ── Main loop ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(_WELCOME_BANNER)

    chat_history   = []
    session_memory = SessionMemory()

    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            print("\nAssistant:\nSession ended.")
            break
        except KeyboardInterrupt:
            print("\nAssistant:\nSession interrupted.")
            break

        if user_input.lower() in ["quit", "exit"]:
            # Print session metrics on clean exit
            try:
                from app.observability.metrics import collector
                collector.print_summary()
            except Exception:
                pass
            break

        # ── Step 1: Collect name ──────────────────────────────────────────
        if session_memory.get("awaiting_name"):
            session_memory.set("name", user_input.strip())
            session_memory.set("awaiting_name", False)
        if _prompt_if_missing(session_memory, "name", "Please share your name."):
            session_memory.set("awaiting_name", True)
            continue

        # ── Step 2: Collect age ───────────────────────────────────────────
        if session_memory.get("awaiting_age"):
            age_val = _try_parse_age(user_input)
            if age_val is not None:
                session_memory.set("age", age_val)
                session_memory.set("awaiting_age", False)
            else:
                print("\nAssistant:")
                print("Please provide your age as a number (e.g., 29).")
                print("\n" + "=" * 50 + "\n")
                continue
        if _prompt_if_missing(session_memory, "age", "Please share your age."):
            session_memory.set("awaiting_age", True)
            continue

        # ── Step 3: Collect & confirm location ───────────────────────────
        if session_memory.get("confirm_location"):
            answer = user_input.strip().lower()
            if answer in ["yes", "y", "correct"] or answer.startswith("y"):
                cand = session_memory.get("location_candidate")
                if isinstance(cand, dict):
                    session_memory.set("location", cand)
                session_memory.set("confirm_location", False)
                print("\nAssistant:")
                print("✅  Location confirmed!\n")
                print("Please describe your disease or main symptoms.")
                print('Example: "I have sudden shortness of breath and chest pain"')
                print("\n" + "=" * 50 + "\n")
                continue
            elif answer in ["no", "n", "incorrect"] or answer.startswith("n"):
                session_memory.set("confirm_location", False)
                session_memory.set("awaiting_location", True)
                print("\nAssistant:")
                print("Please provide a nearby landmark or more specific location.")
                print("\n" + "=" * 50 + "\n")
                continue
            else:
                print("\nAssistant:")
                print("Please answer with yes or no to confirm your location.")
                print("\n" + "=" * 50 + "\n")
                continue

        if session_memory.get("awaiting_location"):
            print("\nAssistant:")
            print("Trying to locate your area...")
            loc_result = location_intake_agent({
                "user_input":    user_input,
                "session_memory": session_memory.data,
                "chat_history":  chat_history,
            })
            if loc_result.get("session_memory"):
                session_memory.update(loc_result["session_memory"])
            if loc_result.get("confirm_location"):
                cand      = loc_result.get("location_candidate") or {}
                formatted = cand.get("formatted") or cand.get("text")
                print(f"Is this your location? {formatted} (yes/no)")
                print("\n" + "=" * 50 + "\n")
                continue
            if loc_result.get("need_location"):
                print(loc_result.get("location_prompt", "Please provide your location."))
                print("\n" + "=" * 50 + "\n")
                continue

        if _prompt_if_missing(
            session_memory, "location",
            "Please share your current location (city, neighbourhood, or a nearby landmark).",
        ):
            session_memory.set("awaiting_location", True)
            continue

        # ── Step 4: Hospital selection ────────────────────────────────────
        if session_memory.get("awaiting_hospital_selection"):
            shown_hospitals = session_memory.get("shown_hospitals") or []
            matched = _match_hospital(user_input, shown_hospitals)

            if matched:
                hospital_name = matched.get("name", "")
                top_disease   = session_memory.get("last_top_disease", "")
                location_data = session_memory.get("location") or {}
                location_hint = (
                    location_data.get("formatted")
                    or location_data.get("text")
                    or ""
                )

                print("\nAssistant:")
                print(f"🔍  Searching for doctors and details at {hospital_name}...")
                print("\n" + "=" * 50 + "\n")

                try:
                    details = hospital_detail_agent(hospital_name, top_disease, location_hint)
                    pretty  = format_hospital_details(hospital_name, top_disease, details)
                    print("\nAssistant:\n")
                    print(pretty)
                except Exception as e:
                    print(f"Could not retrieve hospital details: {e}")

                # Keep selection open in case user wants another hospital
                session_memory.set("awaiting_hospital_selection", True)
                print("\nAssistant:")
                print("─" * 52)
                print("  💡  Select another hospital (name or number 1–5),")
                print("      or type your next medical concern.")
                print("─" * 52)
                print("\n" + "=" * 50 + "\n")
                continue

            elif _looks_like_hospital_query(user_input):
                # User is clearly trying to pick a hospital but the name didn't match
                suggestion = _best_fuzzy_hospital(user_input, shown_hospitals)
                print("\nAssistant:")
                print(f"  ⚠️   I couldn't find \"{user_input.strip()}\" in the hospital list.")
                if suggestion:
                    print(f"  Did you mean: {suggestion['name']}?")
                print("  Please type the exact name or use a number (1–5):")
                for i, h in enumerate(shown_hospitals, 1):
                    print(f"    {i}.  {h.get('name', '')}")
                print("\n" + "=" * 50 + "\n")
                continue

            else:
                # User moved on to a new medical concern — clear the flag
                session_memory.set("awaiting_hospital_selection", False)

        # ── Empty input guard ─────────────────────────────────────────────
        if not user_input.strip():
            print("\nAssistant:")
            print("Please describe your disease or main symptoms.")
            print('Example: "I have sudden shortness of breath and chest pain"')
            print("\n" + "=" * 50 + "\n")
            continue

        # ── Step 5: Run the graph ─────────────────────────────────────────
        result = run_agentic_system(user_input, chat_history, session_memory)

        # Persist diagnosis context for follow-up handling
        _store_diagnosis_context(session_memory, result)

        # Update session memory from graph output
        if result.get("session_memory"):
            session_memory.update(result["session_memory"])

        # Add to chat history
        chat_history.append({"role": "user",      "content": user_input})
        chat_history.append({"role": "assistant", "content": json.dumps(result)})

        # ── Step 6: Display results ───────────────────────────────────────
        try:
            formatted = format_medical_response(result)
            print("\nAssistant:\n")
            print(formatted.get("pretty_text", ""))
        except Exception:
            print("\nAssistant (raw):")
            print(json.dumps(result, indent=2))

        # ── Step 7: Hospital selection prompt ─────────────────────────────
        hospitals = result.get("hospitals") or []
        followup  = result.get("followup_answer")

        if hospitals and not followup:
            # Store hospital list for potential selection next turn
            session_memory.set("shown_hospitals", hospitals)
            session_memory.set("awaiting_hospital_selection", True)
            print("\nAssistant:")
            print("─" * 52)
            print("  💡  Would you like more details about any hospital?")
            print("      Type the hospital name or its number (1–5).")
            print("      Or just describe your next concern to continue.")
            print("─" * 52)

        print("\n" + "=" * 50 + "\n")
