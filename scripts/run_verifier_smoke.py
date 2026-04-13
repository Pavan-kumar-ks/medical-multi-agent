from app.main import run_agentic_system
from app.agents.verifier import verifier_agent
import json

if __name__ == '__main__':
    out = run_agentic_system("i have headache", [])
    print(json.dumps(out, indent=2, ensure_ascii=False))
    ver = verifier_agent(out)
    print('\nVERIFIER REPORT:\n')
    # verifier returns a Pydantic model; serialize to dict for JSON
    print(json.dumps(ver.dict(), indent=2, ensure_ascii=False))
