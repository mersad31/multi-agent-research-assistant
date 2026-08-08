from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from api.streaming import stream_graph_events
from src.graph.builder import build_graph

from src.config.settings import settings


app = FastAPI(title="Research Agent API")

graph = build_graph(interrupt_before=None)

# review_graph = build_graph(interrupt_before=["publisher"])


class ChatRequest(BaseModel):
    query: str

def build_input_state(query: str) -> dict:
    return {
        "messages": [],
        "query": query,
        "sub_tasks": [],
        "sources": [],
        "findings": [],
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
        }
    }


@app.get("/")
async def root():
    return {"messages":"Research Agent API is running."}


@app.post("/chat")
async def chat(request: ChatRequest):
    input_state = build_input_state(request.query)
    config = build_config()

    result = await graph.ainvoke(input_state, config)

    current_node = result.get("current_node")
    if current_node is not None and hasattr(current_node, "value"):
        current_node = current_node.value

    return {
        "query": request.query,
        "report": result.get("report"),
        "sources": result.get("sources", []),
        "sub_tasks": result.get("sub_tasks", []),
        "errors": result.get("errors", []),
        "review_feedback": result.get("review_feedback"),
        "is_sufficient": result.get("is_sufficient"),
        "current_node": result.get("current_node"),
    }


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    input_state = build_input_state(request.query)
    config = build_config()

    return StreamingResponse(
        stream_graph_events(graph, input_state, config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )