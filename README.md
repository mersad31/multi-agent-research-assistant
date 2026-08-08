<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/LangGraph-0.2+-green?logoColor=white" alt="LangGraph" />
  <img src="https://img.shields.io/badge/LangChain-0.3+-teal?logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-orange?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="MIT" />
</p>

<h1 align="center">Multi-Agent Research Assistant</h1>

<p align="center">
  An intelligent, multi-agent research system powered by <strong>LangGraph</strong> that decomposes user queries, searches the web, reviews findings, and produces structured Markdown reports.
</p>

---

## Architecture

The system orchestrates **four specialized agents** in a directed graph with conditional retry loops:

```
                    ┌──────────────┐
                    │    START     │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   PLANNER    │  Decomposes query into sub-tasks
                    └──────┬───────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │      RESEARCHER         │◄─────────────┐
              │  Executes web searches  │              │
              └────────────┬────────────┘              │
                           │                           │
                           ▼                           │
                    ┌──────────────┐                   │
                    │   REVIEWER   │───────────────────┘
                    │  Quality gate │  (if insufficient, retry)
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  PUBLISHER   │  Synthesizes final Markdown report
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │     END      │
                    └──────────────┘
```

### Agent Responsibilities

| Agent | Role | LLM | Output |
|-------|------|-----|--------|
| **Planner** | Breaks down the user query into focused sub-tasks with searchable queries | `MODEL_NAME` | `List[SubTask]` |
| **Researcher** | Iterates over sub-tasks, invokes the search tool, and collects structured findings | No LLM (tool-only) | `List[ResearchFinding]` |
| **Reviewer** | Evaluates the quality and completeness of gathered sources | `MODEL_NAME` | `ReviewResult` (sufficient? feedback?) |
| **Publisher** | Synthesizes all findings into a polished Markdown report | `PUBLISHER_MODEL_NAME` | Markdown string |

### Key Design Decisions

- **State** uses plain `dict` and `str` types (not Pydantic objects) for JSON-serializable checkpointing via `MemorySaver`.
- **Structured Output** (`with_structured_output`) is used at agent boundaries to guarantee type-safe LLM responses.
- **Conditional edges** let the Reviewer route back to the Researcher if findings are insufficient, with a configurable `MAX_RETRIES` guard.
- **Human-in-the-Loop** is supported via `interrupt_before=["publisher"]` for review-before-publish workflows.
- **Dual SSE streaming** delivers both node-level events and token-level streaming for the Publisher agent.
- **Dependency Inversion** via ABC + Factory Pattern for search providers (Tavily / Mock).

---

## Project Structure

```
multi-agent-research-assistant/
├── api/
│   ├── main.py              # FastAPI app with /chat and /chat/stream endpoints
│   └── streaming.py          # SSE event + token streaming utilities
├── src/
│   ├── agents/
│   │   ├── planner.py        # Query decomposition agent
│   │   ├── researcher.py     # Web search execution agent
│   │   ├── reviewer.py       # Quality review agent (conditional router)
│   │   └── publisher.py      # Report synthesis agent
│   ├── config/
│   │   └── settings.py       # Pydantic Settings (env-based configuration)
│   ├── graph/
│   │   ├── builder.py        # StateGraph construction with parameterized interrupt
│   │   └── router.py         # Conditional edge: reviewer → researcher or publisher
│   ├── models/
│   │   └── schemas.py        # Pydantic models for structured LLM output
│   ├── state/
│   │   └── graph_state.py    # TypedDict state definition with reducers
│   └── tools/
│       ├── exceptions.py     # Custom ToolInvocationError
│       └── search.py         # ABC SearchProvider → Tavily / Mock + @tool
├── scripts/
│   └── test_graph.py         # Manual integration test (two-phase HITL)
├── tests/
│   ├── test_state.py         # State schema and reducer tests
│   └── test_tools.py         # MockSearchProvider and error formatting tests
├── .env.example              # Environment variable template
├── .gitignore
└── pyproject.toml            # Project metadata and dependencies
```

---

## Prerequisites

- **Python** >= 3.11
- **OpenAI API Key** ([get one here](https://platform.openai.com/api-keys))
- **Tavily API Key** ([get one here](https://tavily.com/)) — only needed for real web search

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

# 3. Install dependencies (editable mode)
pip install -e ".[dev]"

# 4. Configure environment variables
cp .env.example .env
```

Then edit `.env` and add your API keys:

```env
OPENAI_API_KEY=sk-proj-...
TAVILY_API_KEY=tvly-...
```

---

## Configuration

All configuration is managed through environment variables (loaded from `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required. Your OpenAI API key. |
| `TAVILY_API_KEY` | — | Required for real search. Not needed when `USE_MOCK=True`. |
| `MODEL_NAME` | `gpt-4o-mini` | Model used by Planner and Reviewer. |
| `PUBLISHER_MODEL_NAME` | `gpt-4o` | Model used by Publisher for report generation. |
| `MAX_RETRIES` | `2` | Max Researcher↔Reviewer retry cycles. |
| `USE_MOCK` | `True` | Use `MockSearchProvider` instead of Tavily (no API key needed). |

---

## Usage

### Run Unit Tests

```bash
pytest
```

### Run Integration Test (with Mock Search)

This runs the full graph with mock search results and a two-phase Human-in-the-Loop flow:

```bash
python -m scripts.test_graph
```

### Start the API Server

```bash
uvicorn api.main:app --reload
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/chat` | Synchronous research (returns full report) |
| `POST` | `/chat/stream` | Streaming research via Server-Sent Events (SSE) |

**Example — Synchronous request:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What materials are used in gasoline engine manufacturing?"}'
```

**Example — Streaming request:**

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What materials are used in gasoline engine manufacturing?"}'
```

SSE event format:

```json
{"type": "event", "node": "planner", "message": "planner running..."}
{"type": "token", "node": "publisher", "token": "## "}
{"type": "token", "node": "publisher", "token": "Gasoline"}
{"type": "done"}
```

---

## Human-in-the-Loop

The graph supports pausing before the Publisher node, allowing a human to inspect the gathered sources and provide feedback before the final report is generated.

```python
from src.graph.builder import build_graph

# Enable HITL by specifying which nodes to pause before
graph = build_graph(interrupt_before=["publisher"])

# Phase 1: Run until interrupt
result = graph.invoke(initial_state, config)
# → inspect result["sources"], result["review_feedback"]

# Phase 2: Resume (pass None to continue from checkpoint)
final_result = graph.invoke(None, config)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph (StateGraph, MemorySaver) |
| LLM Integration | LangChain Core + LangChain OpenAI |
| Search | Tavily Python SDK |
| API Framework | FastAPI + Uvicorn |
| Configuration | pydantic-settings |
| Validation | Pydantic v2 |
| Testing | pytest |

---

## License

MIT
