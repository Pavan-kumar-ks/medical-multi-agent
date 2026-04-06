import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import run_agentic_system
import json

scenarios = [
    {
        "name": "Clear-Cut Dengue Case",
        "input": "I have a high fever, my body is aching all over, and a blood test showed I have low platelets."
    },
    {
        "name": "Vague Symptoms",
        "input": "I've been feeling tired and have a headache for a few days."
    },
    {
        "name": "Emergency Symptoms (Cardiac)",
        "input": "I'm experiencing severe chest pain and I'm short of breath."
    },
    {
        "name": "Common Cold Symptoms",
        "input": "I have a runny nose, I'm sneezing a lot, and my throat is sore."
    },
    {
        "name": "Cardiac Arrest Re-test",
        "input": "A person has had a sudden collapse, is unresponsive with no pulse, and is only gasping."
    },
    {
        "name": "Non-Medical Query",
        "input": "What is the capital of France?"
    }
]

def run_tests():
    all_results = {}
    for scenario in scenarios:
        print(f"--- RUNNING SCENARIO: {scenario['name']} ---")
        print(f"Input: {scenario['input']}")
        
        result = run_agentic_system(scenario['input'])
        
        print("\n--- OUTPUT ---")
        # Pretty print the JSON output
        if result:
            print(json.dumps(result, indent=2))
        print("\n" + "="*50 + "\n")
        all_results[scenario['name']] = result
    
    return all_results

if __name__ == "__main__":
    run_tests()
