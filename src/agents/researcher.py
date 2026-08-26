from __future__ import annotations

import json
from typing import List

from src.models.schemas import ResearchFinding
from src.state.graph_state import GraphState, NodeStatus
from src.tools.search import build_search_tool

from src.config.settings import settings


def researcher(state: GraphState) -> dict:
    sub_tasks = state.get("current_sub_tasks", [])
    if not sub_tasks:
        return {
            "errors": ["No sub-tasks found to research."],
            "current_node": NodeStatus.RESEARCHER.value
        }

    search_tool = build_search_tool(use_mock=settings.USE_MOCK, api_key=settings.TAVILY_API_KEY)

    new_findings: List[ResearchFinding] = []
    node_errors: List[str] = []

    for task in sub_tasks:
        search_query = task["search_query"]

        try:
            raw_result = search_tool.invoke({"query": search_query})

            if isinstance(raw_result, str) and (
                    raw_result.startswith("An unexpected")
                    or raw_result.startswith("Error executing")
            ):
                node_errors.append(
                    f"Search failed for '{search_query}': {raw_result}"
                )
                continue
            try:
                search_results = json.loads(raw_result)
            except json.JSONDecodeError:
                node_errors.append(
                    f"Search failed for '{search_query}': {raw_result}"
                )
                continue
            if not isinstance(search_results, list):
                node_errors.append(
                    f"Search failed for '{search_query}': tool returned invalid JSON payload."
                )
                continue

            for res in search_results:
                finding = ResearchFinding(
                    title=res.get("title", "No Title"),
                    url=res.get("url", "https://unknown.com"),
                    content=res.get("content", ""),
                    score=float(res.get("score", 0.0)),
                    query=search_query
                )
                new_findings.append(finding)

        except Exception as exc:
            node_errors.append(
                f"Failed to parse tool output for query: '{search_query}': {exc}"
            )

    return {
        "sources": [f.model_dump() for f in new_findings],
        "current_node": NodeStatus.RESEARCHER.value,
        "errors": node_errors
    }




