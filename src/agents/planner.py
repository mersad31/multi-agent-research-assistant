from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.models.schemas import PlanOutput
from src.state.graph_state import GraphState, NodeStatus

from src.config.settings import settings



_PLANNER_SYSTEM_PROMPT = """
You are a research planning manager.

Your job is to break a user's research request into a small set of focused, executable sub-tasks.
Each sub-task must include:
1. a short description of the research objective
2. a focused search_query that can be sent directly to a web search tool

Rules:
- The search_query must be specific, concrete, and optimized for retrieval.
- Do not write vague, broad, or conversational search queries.
- Decompose the task only as much as necessary.
- If the user's request is simple and does not need decomposition, create exactly one sub-task.
- Keep sub-tasks non-overlapping and useful for downstream research.
- Return reasoning that briefly explains your decomposition strategy.
"""

_PLANNER_PROMPT = ChatPromptTemplate.from_messages([
        ("system", _PLANNER_SYSTEM_PROMPT.strip()),
        ("human", "User Request: {query}"),
    ])


_llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    temperature=0
)
_planner_chain = _PLANNER_PROMPT | _llm.with_structured_output(PlanOutput)


def planner (state: GraphState) -> dict:
    query = state.get("query")
    if not query:
        return {"errors": ["No query found in state."]}

    try:
        plan: PlanOutput = _planner_chain.invoke({"query": query})
        sub_tasks_dump = [sub.model_dump() for sub in plan.sub_tasks]

        return {
            "sub_tasks": sub_tasks_dump,
            "current_sub_tasks": sub_tasks_dump,
            "current_node": NodeStatus.PLANNER.value,
            "errors": [],
        }
    except Exception as e:
        return {
            "errors": [f"Planner node failed: {str(e)}"],
            "current_node": NodeStatus.PLANNER.value
        }