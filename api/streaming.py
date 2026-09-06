from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

PUBLISHER_NODE = "publisher"

_NODE_MESSAGES = {
    "planner": "planner running...",
    "researcher": "researching...",
    "reviewer": "reviewer running...",
    "publisher": "publisher writing report...",
}


def _to_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _extract_chunk_text(chunk: Any) -> str:
    if chunk is None:
        return ""

    content = getattr(chunk, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))

        return "".join(parts)

    return ""


def _get_node_name(event: dict[str, Any]) -> str | None:
    metadata = event.get("metadata") or {}

    langgraph_node = metadata.get("langgraph_node")
    if langgraph_node:
        return str(langgraph_node)

    name = event.get("name")
    if name:
        return str(name)

    return None


def _status_message(node_name: str) -> str:
    return _NODE_MESSAGES.get(node_name, f"{node_name} running...")


async def stream_graph_events(
        graph: Any,
        input_state: dict[str, Any],
        config: dict[str, Any] | None = None
) -> AsyncGenerator[str, None]:
    try:
        async for event in graph.astream_events(input_state, config, version="v2"):
            event_type = event.get("event")
            node_name = _get_node_name(event)

            if event_type == "on_chain_start" and node_name in _NODE_MESSAGES:
                yield _to_sse(
                    {
                        "type": "event",
                        "node": node_name,
                        "message": _status_message(node_name),
                    }
                )

            elif event_type == "on_chat_model_stream":
                # Only stream final report tokens. Planner/reviewer may also call LLMs.
                if node_name != PUBLISHER_NODE:
                    continue

                chunk = event.get("chunk")
                token = _extract_chunk_text(chunk)

                if token:
                    yield _to_sse(
                        {
                            "type": "token",
                            "node": node_name,
                            "token": token,
                        }
                    )

    except Exception: # noqa: BLE001 — a mid-stream failure must degrade to an SSE error event, not crash the generator
        yield _to_sse({
            "type": "error",
            "message": "An internal error occurred while processing your request."
        })
        return

    yield _to_sse({"type": "done"})