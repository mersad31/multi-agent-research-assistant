<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/LangGraph-0.2+-green?logoColor=white" alt="LangGraph" />
  <img src="https://img.shields.io/badge/LangChain-0.3+-teal?logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-orange?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/RAGAS-Evaluation-purple" alt="RAGAS" />
  <img src="https://img.shields.io/badge/Langfuse-Observability-black" alt="Langfuse" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="MIT" />
  <img src="https://github.com/mersad31/multi-agent-research-assistant/actions/workflows/ci.yml/badge.svg" alt="CI" />
</p>

<h1 align="center">Multi-Agent Research Assistant</h1>

<p align="center">
  A production-grade multi-agent research system built on <strong>LangGraph</strong>. It decomposes a user query,
  searches the web, critiques its own findings, replans when evidence is weak, supports human approval before
  publishing, and is continuously evaluated with <strong>RAGAS</strong> metrics streamed to <strong>Langfuse</strong>.
</p>

<p align="center">
  🚀 <strong><a href="https://multi-agent-research-assistant-envw.onrender.com/docs">Live demo (Swagger UI)</a></strong>
  — hosted on Render's free tier; the first request after a period of inactivity can take up to ~50s to wake up.
</p>

---

## Why this project

Chaining a few prompts together is easy. Running that chain in production — with retries that actually improve
results, errors that never crash the process, a human able to intervene mid-run, and a way to know whether a prompt
change made things better or worse — is the harder, senior-level problem this project is built to demonstrate.

## Architecture

Five specialized agents are orchestrated as a LangGraph `StateGraph` with a self-correcting retry loop and an
optional human-approval gate before publishing.

```
                    ┌──────────────┐
                    │    START     │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │   PLANNER    │  Decomposes the query into sub-tasks
                    └──────┬───────┘
                           ▼
              ┌─────────────────────────┐
              │       RESEARCHER        │◄────────────────┐
              │   Executes web searches │                  │
              └────────────┬────────────┘                  │
                           ▼                               │
                    ┌──────────────┐                        │
                    │   REVIEWER   │                        │
                    │ Quality gate │                        │
                    └──────┬───────┘                        │
                 insufficient │ sufficient                  │
                           ▼                                │
                    ┌──────────────┐                        │
                    │  REPLANNER   │────────────────────────┘
                    │ New sub-tasks │  (targets the reviewer's feedback,
                    │  from feedback│   never repeats old queries)
                    └──────────────┘
                           │
                    (sufficient / max retries reached)
                           ▼
              ┌─────────────────────────────┐
              │  Optional human approval    │  interrupt_before=["publisher"]
              │  approve / reject / revise  │  via POST /chat/resume
              └────────────┬─────────────────┘
                           ▼
                    ┌──────────────┐
                    │  PUBLISHER   │  Synthesizes the final Markdown report
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │     END      │
                    └──────────────┘
```

### Agent responsibilities

| Agent | Role | LLM | Output |
|-------|------|-----|--------|
| **Planner** | Breaks the user query into focused, searchable sub-tasks | `MODEL_NAME` | `List[SubTask]` |
| **Researcher** | Runs the search tool for the *current* round of sub-tasks and collects structured findings | tool-only | `List[ResearchFinding]` |
| **Reviewer** | Judges whether the gathered sources are sufficient; decides retry vs. publish | `MODEL_NAME` | `ReviewResult` |
| **Replanner** | Reads the reviewer's feedback and the full history of previously-searched queries, then generates a *new*, non-overlapping set of sub-tasks | `MODEL_NAME` | `List[SubTask]` |
| **Publisher** | Synthesizes all findings into a cited, professional Markdown report | `PUBLISHER_MODEL_NAME` | Markdown string |

### Key design decisions

- **Two-tier task state** — `sub_tasks` (full history, accumulates via `operator.add`) is kept separate from
  `current_sub_tasks` (only this round's work), so the Researcher never re-searches a query it already tried, and
  a retry loop makes genuine progress instead of spinning.
- **Reducer-based state everywhere** — `sources`, `sub_tasks`, and `errors` are all `Annotated[list, operator.add]`,
  so every node returns *only its own contribution* and LangGraph merges the rest. No manual "read the old list,
  append, write it back" bookkeeping inside agents.
- **Defense in depth against wasted LLM calls** — before spending a model call, the Reviewer short-circuits on an
  empty `sources` list or an exhausted retry budget, and the Publisher short-circuits on empty `sources`. A broken
  upstream tool degrades to a clear error message, not a crash or a silently wrong report.
- **Human-in-the-loop, done properly** — `interrupt_before=["publisher"]` pauses the graph right before the final
  report is written. `POST /chat/resume` supports three actions: `approve` (continue unchanged), `reject`
  (cancel without ever running the Publisher), and `revise` (inject human feedback via
  `graph.update_state(..., as_node="reviewer")`, which re-triggers the Replanner loop with the human's note).
- **Per-request Langfuse tracing** — every `/chat`, `/chat/stream`, and `/chat/resume` call gets its own
  `CallbackHandler`, so traces never bleed across concurrent users.
- **Continuous evaluation** — `RAGAS` (`faithfulness`, `answer_relevancy`, plus `context_precision` /
  `context_recall` when ground truth is available) scores every evaluation run and pushes the scores back onto the
  matching Langfuse trace, closing the loop between "a prompt changed" and "did quality go up or down."
- **Dependency inversion for search** — an `ABC SearchProvider` behind a factory lets `USE_MOCK=True` swap in a
  zero-cost mock provider for local development and CI, with the same interface as the real Tavily client.

---

## Project structure

```
multi-agent-research-assistant/
├── api/
│   ├── main.py                    # /chat, /chat/stream, /chat/resume (HITL), global error handler
│   ├── streaming.py                # SSE: node-level events + Publisher token streaming
│   └── evaluation.py               # /evaluate, /evaluate/batch
├── src/
│   ├── agents/
│   │   ├── planner.py               # Initial query decomposition
│   │   ├── replanner.py             # Feedback-driven re-decomposition (retry loop)
│   │   ├── researcher.py            # Web search execution
│   │   ├── reviewer.py              # Quality gate / conditional router source
│   │   └── publisher.py             # Final report synthesis
│   ├── config/settings.py           # Pydantic Settings (env-based configuration)
│   ├── graph/
│   │   ├── builder.py                # StateGraph wiring, parameterized interrupt_before
│   │   └── router.py                 # reviewer → replanner | publisher
│   ├── models/schemas.py            # Pydantic models for structured LLM output
│   ├── state/graph_state.py         # TypedDict state + reducers
│   ├── tools/
│   │   ├── exceptions.py             # ToolInvocationError
│   │   └── search.py                 # ABC SearchProvider → Tavily / Mock + @tool
│   └── evaluation/
│       ├── evaluate_pipeline.py       # Standalone eval runner (single + batch)
│       ├── evaluators/ragas_evaluator.py  # Shared RAGAS + Langfuse scoring logic
│       └── evaluations/golden_dataset.json
├── scripts/
│   └── test_graph.py                # Manual, no-API graph run with a HITL pause/resume walkthrough
├── tests/
│   ├── test_state.py
│   └── test_tools.py
├── .env.example
├── .gitignore
└── pyproject.toml
```

---

## Prerequisites

- **Python** >= 3.11
- **OpenAI API key** ([get one here](https://platform.openai.com/api-keys))
- **Tavily API key** ([get one here](https://tavily.com/)) — only needed when `USE_MOCK=False`
- **Langfuse account** ([cloud.langfuse.com](https://cloud.langfuse.com)) — optional, only needed for tracing and evaluation scoring

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/multi-agent-research-assistant.git
cd multi-agent-research-assistant

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -e ".[dev]"          # core + tests
pip install -e ".[eval]"         # + RAGAS / Langfuse / datasets, only if you'll run evaluation

# 4. Configure environment variables
cp .env.example .env
```

Then edit `.env`:

```env
OPENAI_API_KEY=sk-proj-...
TAVILY_API_KEY=tvly-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

> **Note:** these libraries read `OPENAI_API_KEY` / `TAVILY_API_KEY` from whatever your process environment
> resolves to. A stray system-level environment variable with the same name will silently take priority over
> `.env`. If you see authentication errors despite a correct `.env`, check `echo $OPENAI_API_KEY` (or the Windows
> Environment Variables panel) for a leftover value first.

---

## Configuration

All configuration is managed through environment variables (loaded from `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required. Used by all agent LLM calls. |
| `TAVILY_API_KEY` | — | Required for real search. Not needed when `USE_MOCK=True`. |
| `MODEL_NAME` | `gpt-4o-mini` | Model for Planner, Replanner, and Reviewer. |
| `PUBLISHER_MODEL_NAME` | `gpt-4o` | Model for the Publisher's final report. |
| `MAX_RETRIES` | `2` | Max Researcher ↔ Reviewer/Replanner retry cycles before forcing publish. |
| `USE_MOCK` | `True` | Use `MockSearchProvider` instead of Tavily (no API key needed). |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | — / — / `https://cloud.langfuse.com` | Optional. If unset, the app logs a warning at startup and runs with tracing disabled — it does not fail to start. |
| `EVAL_MODEL` | `gpt-4o` | Judge model used by RAGAS during evaluation. |

---

## Usage

### Run unit tests

```bash
pytest
```

### Run the graph directly (no API, manual HITL walkthrough)

`scripts/test_graph.py` invokes the compiled graph straight from Python with
`interrupt_before=["publisher"]`, prints the gathered sources and review feedback when it pauses, waits for you to
press Enter, then resumes and prints the final report. Useful for debugging the graph itself without going through
FastAPI:

```bash
python -m scripts.test_graph
```

### Start the API server

```bash
uvicorn api.main:app --reload
```

**Endpoints**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/chat` | Run the full research pipeline synchronously |
| `POST` | `/chat/stream` | Same pipeline, streamed as Server-Sent Events |
| `POST` | `/chat/resume` | Approve, reject, or revise a paused (HITL) run |
| `POST` | `/evaluate` | Run one query through the graph and score it with RAGAS |
| `POST` | `/evaluate/batch` | Score a list of queries (or the golden dataset) in the background |

**Basic request**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What materials are used in gasoline engine manufacturing?"}'
```

**Streaming request**

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What materials are used in gasoline engine manufacturing?"}'
```

```json
{"type": "event", "node": "planner", "message": "planner running..."}
{"type": "token", "node": "publisher", "token": "## "}
{"type": "token", "node": "publisher", "token": "Gasoline"}
{"type": "done"}
```

---

## Human-in-the-loop

Set `require_approval: true` on `/chat` to pause the graph right before the Publisher node runs, so a human can
inspect the gathered sources before the final report is written.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What materials are used in gasoline engine manufacturing?", "require_approval": true}'
```

The response comes back with `"status": "paused"` and the `thread_id` needed to resume. Then, from
`POST /chat/resume`:

```bash
# Approve — publish exactly what was gathered
curl -X POST http://localhost:8000/chat/resume \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "<thread_id>", "action": "approve"}'

# Reject — cancel; the Publisher never runs
curl -X POST http://localhost:8000/chat/resume \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "<thread_id>", "action": "reject"}'

# Revise — feed human feedback back into the Replanner loop
curl -X POST http://localhost:8000/chat/resume \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "<thread_id>", "action": "revise", "note": "Also cover EV powertrain materials."}'
```

`revise` may pause again (if the reviewer is still unsatisfied after the new research round) — the response's
`"status"` field tells you whether to call `/chat/resume` again or whether it already reached `"completed"`.

---

## Evaluation

Research quality is scored with [RAGAS](https://github.com/explodinggraph/ragas) and pushed to
[Langfuse](https://langfuse.com) as trace-level scores, so a prompt change's effect on quality is measurable, not
just a feeling.

- **Metrics:** `faithfulness` and `answer_relevancy` always run. `context_precision` and `context_recall` are
  added automatically when ground truth is available (single query) or when *every* row in a batch has ground
  truth (batch mode — RAGAS requires the column to be uniformly present).
- **Golden dataset:** `src/evaluation/evaluations/golden_dataset.json` — a set of `{query, ground_truth}` pairs
  used for regression-style batch evaluation.

```bash
# Score a single query
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"query": "What is LangGraph used for?"}'

# Score every query in the golden dataset (runs in the background)
curl -X POST http://localhost:8000/evaluate/batch \
  -H "Content-Type: application/json" \
  -d '{"use_golden_dataset": true}'
```

Or run it standalone, outside the API:

```bash
python -m src.evaluation.evaluate_pipeline
```

If Langfuse credentials are missing or invalid, evaluation still runs and logs results — it just skips sending
scores upstream, rather than failing the whole request.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph (`StateGraph`, `MemorySaver`, `interrupt_before`) |
| LLM integration | LangChain Core + LangChain OpenAI |
| Search | Tavily Python SDK (with a mock provider for local dev) |
| API framework | FastAPI + Uvicorn |
| Evaluation | RAGAS |
| Observability | Langfuse |
| Configuration | pydantic-settings |
| Validation | Pydantic v2 |
| Testing | pytest |

---

## License

MIT
