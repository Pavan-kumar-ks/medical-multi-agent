# Medical Multi-Agent

AI-assisted multi-agent clinical decision support with RAG, emergency triage, panel-based conflict resolution, location-aware hospital recommendations, and hospital doctor-detail lookup.

## What Is Implemented

- Guided CLI onboarding for patient name, age, and location confirmation.
- Medical-domain query gating (`medical` vs `non-medical`).
- Structured intake, triage, diagnosis, verifier loop, risk analysis, and test recommendation.
- 4-role medical panel review with disagreement detection and adjudication.
- Nearby hospital search with diagnosis-aware ranking (`3 aligned + 2 nearest`).
- Follow-up question handling without re-running full complaint pipeline.
- Hospital detail mode: contact/booking info + doctor profile extraction via web search and scraping.

## Current Flow

The graph entry point is now a question classifier with structured logging at each step:

1. **Question Classifier** → Determine if query is medical or follow-up
2. **Follow-up Branch** (if follow-up): `followup_responder` → end (with context)
3. **Medical Branch** (if new complaint):
   - `domain_expert` → Validate medical domain
   - `intake` → Gather symptom details
   - `location` → Get/confirm patient location
   - `triage` → Assess urgency level
4. **Urgency Branch**:
   - Emergency: `emergency_remedy_agent` → `diagnosis_agent`
   - Non-emergency: `diagnosis_agent` directly
5. **Verification**: `verifier_agent` → Schema validation with retry
6. **Panel Review** → 4-role independent analysis + conflict detection:
   - Primary Diagnostician (diagnosis confidence)
   - Skeptical Reviewer (challenge primary)
   - Evidence Auditor (check evidence quality)
   - Safety Triage Lead (override for critical safety)
7. **Routing**: Adjudicator combines panel into final diagnosis
8. **Hospital Finder** → Geospatial lookup with diagnosis-aware ranking
9. **Post-Diagnosis**:
   - `risk_analyzer` → Risk stratification
   - `test_recommender` → Test recommendations
   - End with formatted report

After results are shown, the CLI can accept hospital number/name input to run the `hospital_detail_agent` and show doctor-specific information.

Each node is instrumented with:
- Agent execution timing
- LLM call metrics (tokens, latency, model)
- Structured success/failure logging
- In-memory trace for debugging

## Repository Layout

- `app/main.py`: CLI runner, onboarding flow, session persistence, hospital selection handling.
- `app/orchestrator/`:
  - `graph.py`: Full state machine graph with panel integration and follow-up routing.
  - `router.py`: Post-hospital routing based on panel/diagnosis confidence.
  - `state.py`: Graph state fields (including panel and follow-up fields).
  - `agent_runner.py`: Safe agent invocation with retries, structured logging, and trace collection.
- `app/agents/`: Medical agents and conversation agents.
- `app/agents/panel/`: Panel role agents (primary_diagnostician, skeptical_reviewer, evidence_auditor, safety_triage_lead), conflict_detector, adjudicator.
- `app/observability/`: 
  - `logger.py`: Structured JSON logging with file and console handlers (medorchestrator.jsonl, agent_trace.jsonl).
  - `metrics.py`: Performance metrics collection (latency, token usage, success rates).
- `app/evaluation/`:
  - `harness.py`: Test case runner with metrics collection and reporting.
  - `runner.py`: CLI for evaluation with category filtering and LLM-as-judge option.
  - `test_queries.py`: 60+ standardized test queries across medical categories.
- `app/tools/formatter.py`: Rich terminal rendering for diagnosis, panel summary, hospitals, and hospital details.
- `app/agents/hospital_finder.py`: Staged-radius nearby search and diagnosis-aware ranking.
- `app/agents/hospital_detail_agent.py`: Hospital contact + doctor extraction workflow.
- `app/scraper/`: Scrapy + Playwright doctor scraping package.
- `app/memory/`: Embeddings, vector store, and session memory helpers.
- `scripts/`: Ingestion, maintenance, and test scenario scripts.
- `logs/`: Output logs and metrics (medorchestrator.jsonl, agent_trace.jsonl, metrics.jsonl, eval_report.json).

## LLM, RAG, and Guarded Diagnosis

- LLM wrapper: `app/config.py` (`llm_call`).
- Retrieval: `app/tools/retriever.py` with FAISS-backed vector store.
- Diagnosis output is schema-validated by verifier and retried with verifier feedback.
- If verification still fails, the graph falls back to a safe unknown diagnosis.

## Panel-Based Conflict Resolution

Panel roles used in `app/agents/panel/`:

- Primary Diagnostician
- Skeptical Reviewer
- Evidence Auditor
- Safety Triage Lead

The pipeline then:

- Detects conflicts (`conflict_detector`)
- Adjudicates a final panel decision (`adjudicator`)
- Can trigger emergency override when safety risk is detected
- Feeds panel-adjudicated diagnosis downstream for hospital/risk/test routing

## Hospital and Location Features

- Location normalization and confirmation via `location_intake_agent`.
- Nearby hospitals from OSM stack:
  - Geocoding: Nominatim
  - Nearby hospitals: Overpass
  - Travel distance/time: OSRM
- Radius expansion strategy: `5 km -> 10 km -> 20 km`.
- Ranking strategy: top diagnosis specialty alignment first, then nearest fallbacks.

## Hospital Doctor Details and Scraping

Hospital detail lookup (`hospital_detail_agent`) does:

1. Web-search based extraction of hospital contact/booking info.
2. Doctor scraping using Scrapy spider (`app/scraper/spiders/doctor_spider.py`).
3. Specialty filtering based on diagnosed condition.
4. LLM/web fallback extraction when scraper returns no profiles.

Scraper stack:

- `scrapy`
- `scrapy-playwright`
- Playwright Chromium runtime

## Requirements

Current `requirements.txt` includes:

- `fastapi`, `uvicorn`
- `pydantic`, `python-dotenv`, `requests`
- `groq`
- `faiss-cpu`, `sentence-transformers`
- `duckduckgo-search`
- `scrapy`, `scrapy-playwright`
- `langgraph`

After installing requirements, install the Playwright browser runtime:

```powershell
playwright install chromium
```

## Environment Variables

Required:

- `GROQ_API_KEY`
- `MODEL_NAME`

Common optional map variables:

- `NOMINATIM_URL` (default `https://nominatim.openstreetmap.org/search`)
- `OVERPASS_URL` (default `https://overpass-api.de/api/interpreter`)
- `OSRM_URL` (default `http://router.project-osrm.org/route/v1/driving`)
- `NOMINATIM_EMAIL` (recommended contact in user-agent)

## Setup and Run

1. Create and activate virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
playwright install chromium
```

3. Configure `.env` (minimum):

```env
GROQ_API_KEY=your_key_here
MODEL_NAME=llama-3.3-70b-versatile
```

4. Build vector index (first run):

```powershell
python scripts/ingest_kb.py
```

5. Start CLI:

```powershell
python app/main.py
```

## Observability & Logging

All agent invocations are logged as newline-delimited JSON (JSONL) for observability:

- **`logs/medorchestrator.jsonl`**: Core application logs (DEBUG+) with agent events, LLM calls, token usage
- **`logs/agent_trace.jsonl`**: Dedicated agent state-transition trace logs for graph flow analysis
- **`logs/metrics.jsonl`**: Aggregated performance metrics (latency, token counts, success/error rates)

Access logs via:
```powershell
Get-Content logs/medorchestrator.jsonl | ConvertFrom-Json | Format-Table -Property ts, level, msg, agent
```

## Agent Tracing

The in-memory agent trace (`reset_agent_trace()`, `get_agent_trace()`) tracks:
- Agent names and execution order
- Success/failure with durations
- Attempts and error details
- Token usage per LLM call

Traces are included in API responses under `_agent_trace` field and persisted to logs.

## Evaluation

The evaluation harness supports comprehensive testing:

```powershell
# Run all 60+ test queries across categories
python -m app.evaluation.runner

# Run specific category only
python -m app.evaluation.runner --category cardiac

# Run with LLM-as-judge
python -m app.evaluation.runner --llm-judge

# Limit to N queries
python -m app.evaluation.runner --limit 10

# Custom output
python -m app.evaluation.runner --output custom_report.json
```

Test queries cover:
- Cardiac (10 queries)
- Respiratory (8 queries)
- Neurological (7 queries)
- Gastrointestinal (6 queries)
- Infectious (7 queries)
- Metabolic (4 queries)
- Orthopedic (3 queries)
- Follow-up scenarios (5 queries)
- Non-medical queries (4 queries)
- Landmark queries (3 queries)
- Pediatric (3 queries)

Quick test scenarios:
```powershell
python scripts/run_scenarios.py      # 6 manual test cases
python scripts/run_failure_scenario.py # Fallback handling demo
```

## Recent Changes

**May 2, 2026:**
- Full pipeline execution logging and tracing infrastructure
- Comprehensive LLM call metrics (token counts, latency, model)
- Evaluation framework with 60+ standardized test queries
- In-memory and persistent agent trace for debugging
- Agent runner with retry logic, timing, and structured error handling
- Prompt truncation warnings for oversized contexts
- Follow-up question handling verified through evaluation

**Logs from recent execution:**
- 60 evaluation queries processed successfully
- Agent pipeline: question_classifier → domain_expert → intake → location → triage → emergency/diagnosis → panel → hospital → risk/tests
- Panel agents executing independently: primary diagnostician, skeptical reviewer, evidence auditor, safety triage lead
- Hospital finder with geospatial lookup (~64 seconds for OSM queries)
- Follow-up responses through question classifier + followup_responder

## Notes

- The active runtime experience is CLI-first.
- Follow-up questions are answered from stored diagnosis context.
- Hospital detail mode is triggered by selecting a listed hospital number/name after report output.
- All agent execution is traced and logged for debugging and monitoring.
- Evaluation results are JSON-structured for programmatic analysis.

## Testing

```powershell
# Unit tests
pytest -q

# Run test scenarios
python scripts/run_scenarios.py

# Run evaluation harness
python -m app.evaluation.runner --limit 5
```

## Troubleshooting

- If vector index is missing, run: `python scripts/ingest_kb.py`.
- If hospital scraping returns nothing, ensure Scrapy + Playwright and Chromium are installed.
- If LLM responses fail, verify `GROQ_API_KEY` and `MODEL_NAME`.

