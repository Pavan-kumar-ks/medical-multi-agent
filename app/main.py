import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.orchestrator.graph import build_graph
from app.memory.vector_store import load_vector_store
import json
from app.tools.formatter import format_medical_response
from app.memory.session_memory import SessionMemory
from app.agents.location_intake import location_intake_agent
from app.orchestrator.agent_runner import reset_agent_trace, get_agent_trace

def run_agentic_system(user_input: str, chat_history: list, session_memory: SessionMemory | None = None):
    load_vector_store()
    graph = build_graph()
    # reset per-run trace
    reset_agent_trace()

    result = graph.invoke({
        "user_input": user_input,
        "chat_history": chat_history,
        "session_memory": session_memory.data if session_memory else {}
    })

    # attach agent execution trace (not sensitive)
    try:
        result["_agent_trace"] = get_agent_trace()
    except Exception:
        result["_agent_trace"] = []

    return result

def _prompt_if_missing(session_memory: SessionMemory, key: str, prompt: str) -> bool:
    if not session_memory.get(key):
        print("\nAssistant:")
        print(prompt)
        print("\n" + "="*50 + "\n")
        return True
    return False


def _try_parse_age(text: str):
    try:
        age = int("".join([c for c in text if c.isdigit()]))
        if age > 0:
            return age
    except Exception:
        return None
    return None


if __name__ == "__main__":
    chat_history = []
    session_memory = SessionMemory()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        # Step 1: Collect name
        if session_memory.get("awaiting_name"):
            session_memory.set("name", user_input.strip())
            session_memory.set("awaiting_name", False)
        if _prompt_if_missing(session_memory, "name", "Please share your name."):
            session_memory.set("awaiting_name", True)
            continue

        # Step 2: Collect age
        if session_memory.get("awaiting_age"):
            age_val = _try_parse_age(user_input)
            if age_val is not None:
                session_memory.set("age", age_val)
                session_memory.set("awaiting_age", False)
            else:
                print("\nAssistant:")
                print("Please provide your age as a number (e.g., 29).")
                print("\n" + "="*50 + "\n")
                continue
        if _prompt_if_missing(session_memory, "age", "Please share your age."):
            session_memory.set("awaiting_age", True)
            continue

        # Step 3: Collect location (before diagnosis)
        # Handle confirmation response before attempting new geocoding
        if session_memory.get("confirm_location"):
            if user_input.strip().lower() in ["yes", "y", "correct"]:
                cand = session_memory.get("location_candidate")
                if isinstance(cand, dict):
                    session_memory.set("location", cand)
                session_memory.set("confirm_location", False)
            else:
                # Ask user for a better location hint
                session_memory.set("confirm_location", False)
                session_memory.set("awaiting_location", True)
                print("\nAssistant:")
                print("Please provide a nearby landmark or more specific location.")
                print("\n" + "="*50 + "\n")
                continue

        if session_memory.get("awaiting_location"):
            # Ask the location agent to resolve location
            print("\nAssistant:")
            print("Trying to locate your area...")
            loc_result = location_intake_agent({
                "user_input": user_input,
                "session_memory": session_memory.data,
                "chat_history": chat_history,
            })
            if loc_result.get("session_memory"):
                session_memory.update(loc_result.get("session_memory"))

            if loc_result.get("confirm_location"):
                cand = loc_result.get("location_candidate") or {}
                formatted = cand.get("formatted") or cand.get("text")
                print(f"Is this your location? {formatted} (yes/no)")
                print("\n" + "="*50 + "\n")
                continue

            if loc_result.get("need_location"):
                print(loc_result.get("location_prompt", "Please provide your location."))
                print("\n" + "="*50 + "\n")
                continue
        if _prompt_if_missing(session_memory, "location", "Please share your current location (city, neighborhood, or a nearby landmark)."):
            session_memory.set("awaiting_location", True)
            continue
            
        result = run_agentic_system(user_input, chat_history, session_memory)

        # Add the interaction to the chat history (stored but not printed)
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": json.dumps(result)})

        # Update session memory if returned by the graph
        if result.get("session_memory"):
            session_memory.update(result.get("session_memory"))

        # Format the assistant response for human-friendly display
        try:
            formatted = format_medical_response(result)

            # Print agent execution trace (what ran during this request)
            trace = result.get("_agent_trace", [])
            if trace:
                print("\nAgent execution trace:")
                for t in trace:
                    name = t.get("agent")
                    ok = t.get("ok")
                    attempts = t.get("attempts")
                    print(f" - {name}: ok={ok} attempts={attempts}")

            # Print an easy-to-read summary plus immediate actions
            print("\nAssistant (summary):\n")
            print(formatted.get("pretty_text", ""))

            # Also print the sanitized structured JSON for machine use
            print("\nAssistant (raw JSON):")
            print(json.dumps(formatted, indent=2))
        except Exception:
            # Fallback to raw output if formatting fails
            print("\nAssistant (raw):")
            print(json.dumps(result, indent=2))
        print("\n" + "="*50 + "\n")