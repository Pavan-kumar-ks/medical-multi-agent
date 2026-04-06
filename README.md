# Medical Multi-Agent

Lightweight multi-agent pipeline for clinical decision support (RAG + rule-based).

**Repository layout**
- `app/` — application code
  - `main.py` — CLI entrypoint and programmatic runner (`run_agentic_system`)
  - `config.py` — LLM client wrapper and env config
  - `agents/` — agent implementations (`intake`, `diagnosis`, `risk_analyzer`, `test_recommender`, ...)
  - `orchestrator/` — flow graph and routing logic
  - `tools/` — retrieval and rules engine helpers
  - `memory/` — embeddings and vector store helpers
  - `data/` — vector index and document store produced by ingestion
  - `schemas/` — Pydantic models for structured data
- `scripts/` — helper scripts (e.g., `ingest_data.py`)
- `notebooks/` — experimentation

This README documents the pipeline flow, setup, and how to run the project locally.

**High-level pipeline flow**
- Input: raw free-text patient report provided to the system.
- `intake_agent` (app/agents/intake.py): converts free text into a structured `PatientData` object using an LLM prompt.
- Orchestrator (app/orchestrator/graph.py): builds a state graph with nodes `intake -> diagnosis -> (risk | tests) -> tests` and invokes nodes in sequence.
- `diagnosis_agent` (app/agents/diagnosis.py): performs RAG (retrieval-augmented generation):
  - Uses `retrieve_context` (app/tools/retriever.py) which calls `get_embedding` + `search` on the vector store to gather supporting documents.
  - Calls the LLM (via `app.config.llm_call`) with the patient data + retrieved context and returns a JSON diagnosis output.
- Router (app/orchestrator/router.py): `route_after_diagnosis` inspects `diagnosis` output (top confidence) and decides to route to `risk` (rule-based analysis) or directly to `tests` (when low confidence → request tests).
- `risk_analyzer_agent` (app/agents/risk_analyzer.py): runs deterministic medical rules (`app.tools.rules_engine`) to produce risk flags.
- `test_recommender_agent` (app/agents/test_recommender.py): LLM-backed agent recommending 3–5 tests in JSON form.
- Final output: aggregated `patient`, `diagnosis`, `risks`, and `recommended_tests` returned by the orchestrator.

**Data & memory / RAG**
- Embeddings: `app/memory/embeddings.py` uses `sentence-transformers` (`all-MiniLM-L6-v2`) to compute embeddings.
- Vector store: `app/memory/vector_store.py` stores a FAISS index and a `documents.npy` file under `app/data/`.
- Ingestion: `scripts/ingest_data.py` shows an example pipeline that embeds a small list of documents and writes `vector.index` and `documents.npy` into `app/data/`.
- Runtime: `app.main` calls `load_vector_store()` before invoking the graph so retrieval works.

**Configuration / LLM**
- `app/config.py` wraps the Groq client. Required environment variables:
  - `GROQ_API_KEY` — API key for Groq
  - `MODEL_NAME` — model identifier to call

Create a `.env` file in the repository root or set these variables in your environment.

**Setup & quick run**
Prerequisites: Python 3.9+ recommended.

1) Create a virtual environment and install dependencies:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
# additional runtime deps used by embeddings/vector store
pip install sentence-transformers faiss-cpu numpy
```

2) Create `.env` (example):

```
GROQ_API_KEY=sk-...
MODEL_NAME=groq-1.0
```

3) Build the vector DB (RAG index) before running the pipeline:

```powershell
python scripts/ingest_data.py
```

This will create `app/data/vector.index` and `app/data/documents.npy` used at runtime.

4) Run the CLI runner:

```powershell
python app/main.py
# enter a free-text patient description when prompted
```

Programmatic usage (importable):

```python
from app.main import run_agentic_system
print(run_agentic_system('38yo female with fever and severe body pain for 3 days'))
```

**Notes about running as a server**
- The repository previously included a FastAPI scaffold but the current `app/main.py` is a CLI/programmatic runner. Running `uvicorn app.main:app` is not supported unless you add a FastAPI `app` object.

**Tests**
- Unit tests are present as `test_*.py` files in the repo root. Run them with:

```powershell
pip install pytest
pytest -q
```

**Developer details / important files**
- `app/orchestrator/graph.py` — constructs the `StateGraph` (nodes and edges) and compiles the graph used by `main.py`.
- `app/orchestrator/router.py` — routing policy after diagnosis: inspects diagnosis confidence to choose `risk` vs `tests`.
- `app/agents/intake.py` — input parsing LLM prompt → `PatientData`.
- `app/agents/diagnosis.py` — RAG + LLM diagnosis generation and safe JSON parsing
- `app/tools/retriever.py` — glue between embeddings and FAISS search
- `app/memory/vector_store.py` — FAISS index init/load/search; raises a clear error if index missing (run `scripts/ingest_data.py`).

**Troubleshooting**
- If you get `Vector DB not found. Run ingest_data first.` run `python scripts/ingest_data.py`.
- Embeddings model download may take time on first run; ensure you have network access and sufficient disk space.
- LLM errors: ensure `GROQ_API_KEY` and `MODEL_NAME` are set and valid.

**Next steps / improvements**
- Add a FastAPI wrapper (`app` FastAPI instance) to expose HTTP endpoints.
- Add robust input validation and unit tests for each agent.
- Add CI that builds the FAISS index for tests or mocks retrieval during unit tests.

If you'd like, I can also:
- add a minimal FastAPI server wrapper, or
- update `requirements.txt` to include `sentence-transformers`, `faiss-cpu`, and `numpy`.

---

