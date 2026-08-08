from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.models.schemas import ReviewResult
from src.state.graph_state import GraphState, NodeStatus

from src.config.settings import settings



_REVIEWER_SYSTEM_PROMPT = """
You are a senior research reviewer. 
Your task is to assess if the gathered sources are sufficient to answer the user's query comprehensively.

Evaluation Criteria:
1. Depth: Do the sources cover all aspects of the query?
2. Relevance: Are the findings directly related to the user's intent?
3. Recency: If the query requires current data, is the information up-to-date?

If the information is sufficient, set is_sufficient = True.
If more research is needed, set is_sufficient = False and provide specific feedback on what is missing.
"""

_REVIEWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _REVIEWER_SYSTEM_PROMPT.strip()),
    ("human", "Original Query: {query}\n\nGathered Sources:\n{sources_text}\n\nEvaluate the research quality:")
])

_llm = ChatOpenAI(model=settings.MODEL_NAME, temperature=0)
_reviewer_chain = _REVIEWER_PROMPT | _llm.with_structured_output(ReviewResult)


def reviewer(state: GraphState) -> dict:
    query = state.get("query")
    sources = state.get("sources")
    count_retries = state.get("count_retries")
    max_retries = state.get("max_retries")
    existing_errors = state.get("errors") or []

    if count_retries >= max_retries:
        return {
            "is_sufficient": True,  # Force continue to Publisher
            "review_feedback": "Max retries reached.",
            "errors": existing_errors + ["Max retries reached. Proceeding with existing findings."],
            "current_node": NodeStatus.PUBLISHER
        }

    sources_text = "\n".join([
        f"- {s['title']} ({s['url']}): {s['content'][:200]}..."
        for s in sources
    ])

    try:
        review_output: ReviewResult = _reviewer_chain.invoke({
            "query": query,
            "sources_text": sources_text,
        })

        is_suff = review_output.is_sufficient

        new_errors = existing_errors if is_suff else existing_errors + [review_output.feedback]

        return {
            "is_sufficient": is_suff,
            "review_feedback": review_output.feedback if not is_suff else None,
            "current_node": NodeStatus.REVIEWER,
            "errors": new_errors,
            "count_retries": count_retries + 1 if not is_suff else count_retries
        }

    except Exception as e:

        return {
            "errors": (state.get("errors") or []) + [f"Reviewer failed: {str(e)}"],
            "is_sufficient": True,
            "review_feedback": None,
            "count_retries": count_retries + 1
        }

