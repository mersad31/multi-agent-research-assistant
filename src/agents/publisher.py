from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.state.graph_state import GraphState, NodeStatus

from src.config.settings import settings

_PUBLISHER_SYSTEM_PROMPT = """
You are a professional research writer and publisher.
Your task is to synthesize the gathered research findings into a comprehensive, clear, and well-structured Markdown report.

Guidelines:
1. Address the original user query thoroughly using ONLY the provided sources.
2. Structure the report with clear headings, subheadings, and bullet points.
3. Include inline citations to the sources where applicable (e.g., "[Title](URL)").
4. Maintain a professional, objective, and analytical tone.
5. If the sources contain conflicting information, present both sides objectively.
"""

_PUBLISHER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _PUBLISHER_SYSTEM_PROMPT.strip()),
    ("human", "Original Query: {query}\n\nSources:\n{sources_text}\n\nGenerate the final Markdown report:")
])

_llm = ChatOpenAI(model=settings.PUBLISHER_MODEL_NAME, temperature=0.3)
_publisher_chain = _PUBLISHER_PROMPT | _llm


def publisher(state: GraphState) -> dict:
    query = state.get("query")
    sources = state.get("sources")

    if not sources:
        return {
            "report": "No sources were found to generate a report.",
            "current_node": NodeStatus.PUBLISHER
        }

    sources_text = ""
    for idx, s in enumerate(sources, 1):
        sources_text += f"[{idx}] Title: {s['title']}\nURL: {s['url']}\nContent: {s['content']}\n\n"

    try:
        response = _publisher_chain.invoke({
            "query": query,
            "sources_text": sources_text.strip()
        })

        report_content = response.content if response.content else ""

    except Exception as e:
        report_content = f"# Research Report\n\nAn error occurred while generating the report: {str(e)}"
        return {
            "report": report_content,
            "errors": [f"Publisher failed: {str(e)}"],
            "current_node": NodeStatus.PUBLISHER
        }

    return {
        "report": report_content,
        "current_node": NodeStatus.PUBLISHER
    }