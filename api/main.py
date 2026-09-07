from __future__ import annotations

import logging
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from api.streaming import stream_graph_events
from src.config.settings import settings
from src.graph.builder import build_graph

logger = logging.getLogger(__name__)

app = FastAPI(title="Research Agent API")

try:
    from api.evaluation import router as eval_router
    app.include_router(eval_router)
except ImportError:
    logger.warning(
        "Evaluation dependencies (ragas/datasets) not installed; "
        "/evaluate endpoints are disabled. Install with: pip install -e '.[eval]'"
    )

graph = build_graph(interrupt_before=None)
review_graph = build_graph(interrupt_before=["publisher"])

_langfuse_client = get_client()
try:
    if not _langfuse_client.auth_check():
        logger.warning("Langfuse authentication failed at startup. Traces will not be sent to Langfuse.")
except Exception:  # noqa: BLE001 — a bad/misconfigured Langfuse key must never prevent the server from starting
    logger.warning("Langfuse auth check raised an error at startup. Traces will not be sent to Langfuse.")


class ChatRequest(BaseModel):
    query: str
    thread_id: str | None = None
    require_approval: bool = False


class ResumeRequest(BaseModel):
    thread_id: str
    action: Literal["approve", "reject", "revise"]
    note: str | None = None


def build_input_state(query: str) -> dict:
    return {
        "messages": [],
        "query": query,
        "sub_tasks": [],
        "sources": [],
        "findings": [], # Reserved for a future Coder/Analyst agent (data analysis output). Not populated yet — use `sources` for raw search results.
        "report": "",
        "current_node": None,
        "max_retries": settings.MAX_RETRIES,
        "count_retries": 0,
        "errors": [],
        "is_sufficient": None,
        "review_feedback": None,
    }


def build_config(thread_id: str = "default") -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
        },
        "callbacks": [CallbackHandler()],
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):

    logger.exception("Unhandled exception in request: %s %s", request.method, request.url.path)

    return JSONResponse(
        status_code=500,
        content={"error": "An internal error occurred. Please try again later."}
    )


@app.get("/")
async def root():
    return {"messages":"Research Agent API is running."}


@app.post("/chat")
async def chat(request: ChatRequest):
    thread_id = request.thread_id or str(uuid.uuid4())
    input_state = build_input_state(request.query)
    config = build_config(thread_id)

    selected_graph = review_graph if request.require_approval else graph

    result = await selected_graph.ainvoke(input_state, config)

    state_snapshot = selected_graph.get_state(config)
    is_paused = bool(state_snapshot.next)

    current_node = result.get("current_node")

    if is_paused:
        return {
            "thread_id": thread_id,
            "query": request.query,
            "status": "paused",
            "sources": result.get("sources", []),
            "sub_tasks": result.get("sub_tasks", []),
            "errors": result.get("errors", []),
            "review_feedback": result.get("review_feedback"),
            "is_sufficient": result.get("is_sufficient"),
            "current_node": current_node,
        }

    return {
        "thread_id": thread_id,
        "query": request.query,
        "status": "completed",
        "sources": result.get("sources", []),
        "sub_tasks": result.get("sub_tasks", []),
        "report": result.get("report"),
        "errors": result.get("errors", []),
        "review_feedback": result.get("review_feedback"),
        "is_sufficient": result.get("is_sufficient"),
        "current_node": current_node,
    }


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    thread_id = request.thread_id or str(uuid.uuid4())
    input_state = build_input_state(request.query)
    config = build_config(thread_id)

    return StreamingResponse(
        stream_graph_events(graph, input_state, config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

@app.post("/chat/resume")
async def chat_resume(request: ResumeRequest):
    config = build_config(request.thread_id)

    if request.action == "approve":
        result = await review_graph.ainvoke(None, config)

        current_node = result.get("current_node")

        return {
            "thread_id": request.thread_id,
            "query": result.get("query"),
            "status": "completed",
            "sources": result.get("sources", []),
            "sub_tasks": result.get("sub_tasks", []),
            "report": result.get("report"),
            "errors": result.get("errors", []),
            "review_feedback": result.get("review_feedback"),
            "is_sufficient": result.get("is_sufficient"),
            "current_node": current_node,
        }

    if request.action == "reject":
        review_graph.update_state(
            config,
            {"report": "Report generation was cancelled by user review."},
        )
        state_snapshot = review_graph.get_state(config)
        values = state_snapshot.values

        return {
            "thread_id": request.thread_id,
            "status": "rejected",
            "report": values.get("report"),
            "sources": values.get("sources", []),
        }

    if request.action == "revise":
        if not request.note:
            raise HTTPException(
                status_code=422,
                detail="A 'note' is required when action is 'revise'."
            )

        review_graph.update_state(
            config,
            {
                "is_sufficient": False,
                "review_feedback": request.note,
            },
            as_node="reviewer",
        )

        result = await review_graph.ainvoke(None, config)

        state_snapshot = review_graph.get_state(config)
        is_paused = bool(state_snapshot.next)

        current_node = result.get("current_node")

        if is_paused:
            return {
                "thread_id": request.thread_id,
                "status": "paused",
                "query": result.get("query"),
                "sources": result.get("sources", []),
                "sub_tasks": result.get("sub_tasks", []),
                "report": result.get("report"),
                "errors": result.get("errors", []),
                "review_feedback": result.get("review_feedback"),
                "is_sufficient": result.get("is_sufficient"),
                "current_node": current_node,
            }

        return {
            "thread_id": request.thread_id,
            "status": "completed",
            "query": result.get("query"),
            "sources": result.get("sources", []),
            "sub_tasks": result.get("sub_tasks", []),
            "report": result.get("report"),
            "errors": result.get("errors", []),
            "review_feedback": result.get("review_feedback"),
            "is_sufficient": result.get("is_sufficient"),
            "current_node": current_node,
        }

    raise HTTPException(
        status_code=500,
        detail=f"Unhandled resume action: {request.action}",
    )