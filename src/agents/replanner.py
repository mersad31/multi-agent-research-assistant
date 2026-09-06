from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config.settings import settings
from src.models.schemas import PlanOutput
from src.state.graph_state import GraphState, NodeStatus

_REPLANNER_SYSTEM_PROMPT = """
You are a research re-planning specialist.

You are being called because a previous round of research was judged INSUFFICIENT
by a reviewer. Your job is to generate a new, focused set of sub-tasks that closes
the specific gaps described in the reviewer's feedback.

You will be given:
1. The original user query.
2. The reviewer's feedback explaining what is missing or weak.
3. The list of search queries that were already executed in previous rounds.

Rules:
- Do NOT repeat or trivially reword any of the previously searched queries.
- Every new sub-task must directly address a gap mentioned in the review feedback.
- Keep the new sub-task list small and focused — only what's needed to close the gap,
  not a full re-decomposition of the original query.
- Each sub-task must still include a short description and a focused, web-searchable
  search_query, exactly like the original planning format.
- Return reasoning that explains how each new sub-task addresses the feedback.
"""

_REPLANNER_PROMPT = ChatPromptTemplate.from_messages([
        ("system", _REPLANNER_SYSTEM_PROMPT.strip()),
        ("human", "Original Query: {query}\n\nReview Feedback:\n{review_feedback}\n\nPreviously Searched Queries:\n{previous_sub_tasks}"),
    ])

_llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    temperature=0
)
_replanner_chain = _REPLANNER_PROMPT | _llm.with_structured_output(PlanOutput)

def replanner(state: GraphState) -> dict:
    query = state.get("query")
    review_feedback =  state.get("review_feedback")
    sub_tasks = state.get("sub_tasks")

    if not query:
        return {"errors": ["No query found in state."]}

    previous_tasks_text = "\n".join([
        f"- {t['description']} (query: {t['search_query']})"
        for t in sub_tasks
    ])

    try:
        plan: PlanOutput = _replanner_chain.invoke({
            "query": query,
            "review_feedback": review_feedback,
            "previous_sub_tasks": previous_tasks_text,
        })

        sub_tasks_dump = [sub.model_dump() for sub in plan.sub_tasks]

        return {
            "sub_tasks": sub_tasks_dump,
            "current_sub_tasks": sub_tasks_dump,
            "current_node": NodeStatus.REPLANNER.value,
            "errors": [],
        }

    except Exception as e: # noqa: BLE001 — LLM/tool failures are unpredictable; never crash the graph
        return {
            "errors": [f"Replanner failed: {e!s}"],
            "current_node": NodeStatus.REPLANNER.value
        }