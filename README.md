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

The graph entry point is now a question classifier:

1. `question_classifier`
2. If follow-up: `followup_responder` -> end
3. Else: `domain_expert` -> `intake` -> `location` -> `triage`
4. Emergency branch: `emergency` -> `diagnosis`
5. Non-emergency branch: `diagnosis`
6. `panel` (primary diagnostician, skeptical reviewer, evidence auditor, safety triage lead)
7. `hospital`
8. `route_after_diagnosis` -> `risk` or `tests`
9. `risk` -> `tests` -> end

After results are shown, the CLI can accept hospital number/name input to run the hospital-detail agent and show doctor-specific information.

## Repository Layout

- `app/main.py`: CLI runner, onboarding flow, session persistence, hospital selection handling.
- `app/orchestrator/graph.py`: full state graph, panel integration, follow-up routing.
- `app/orchestrator/router.py`: post-hospital routing based on panel/diagnosis confidence.
- `app/orchestrator/state.py`: graph state fields (including panel and follow-up fields).
- `app/agents/`: medical agents and conversation agents.
- `app/agents/panel/`: panel role agents, conflict detector, adjudicator.
- `app/tools/formatter.py`: rich terminal rendering for diagnosis, panel summary, hospitals, and hospital details.
- `app/agents/hospital_finder.py`: staged-radius nearby search and diagnosis-aware ranking.
- `app/agents/hospital_detail_agent.py`: hospital contact + doctor extraction workflow.
- `app/scraper/`: Scrapy + Playwright doctor scraping package.
- `app/memory/`: embeddings/vector store/session memory helpers.
- `scripts/`: ingestion and maintenance scripts.

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

## Notes

- The active runtime experience is CLI-first.
- Follow-up questions are answered from stored diagnosis context.
- Hospital detail mode is triggered by selecting a listed hospital number/name after report output.

## Testing

```powershell
pytest -q
```

## Troubleshooting

- If vector index is missing, run: `python scripts/ingest_kb.py`.
- If hospital scraping returns nothing, ensure Scrapy + Playwright and Chromium are installed.
- If LLM responses fail, verify `GROQ_API_KEY` and `MODEL_NAME`.

