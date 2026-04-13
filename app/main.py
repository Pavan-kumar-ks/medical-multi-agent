import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.orchestrator.graph import build_graph
from app.memory.vector_store import load_vector_store
import json
from app.tools.formatter import format_medical_response
from app.orchestrator.agent_runner import reset_agent_trace, get_agent_trace

def run_agentic_system(user_input: str, chat_history: list):
    load_vector_store()
    graph = build_graph()
    # reset per-run trace
    reset_agent_trace()

    result = graph.invoke({
        "user_input": user_input,
        "chat_history": chat_history
    })

    # attach agent execution trace (not sensitive)
    try:
        result["_agent_trace"] = get_agent_trace()
    except Exception:
        result["_agent_trace"] = []

    return result

if __name__ == "__main__":
    chat_history = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        result = run_agentic_system(user_input, chat_history)

        # Add the interaction to the chat history (stored but not printed)
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": json.dumps(result)})

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