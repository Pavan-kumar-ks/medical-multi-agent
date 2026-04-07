from app.main import run_agentic_system
import json

if __name__ == "__main__":
    out = run_agentic_system("i have headache", [])
    print(json.dumps(out, indent=2, ensure_ascii=False))
