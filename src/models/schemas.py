from __future__ import annotations

from pydantic import BaseModel, Field


class SubTask(BaseModel):
    description: str = Field(
        ...,
        description="A short description of the sub-task that explains what should be researched.",
    )
    search_query: str = Field(
        ...,
        description=(
            "A focused, web-searchable query that can be sent directly to the search tool. "
            "It should be specific, concrete, and optimized for retrieval, not a vague natural-language goal."
        ),
    )


class PlanOutput(BaseModel):
    sub_tasks: list[SubTask] = Field(
        ...,
        description="A list of sub-tasks created by decomposing the user's main request.",
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why the task was divided this way for transparency and debugging.",
    )


class ResearchFinding(BaseModel):
    title: str = Field(
        ...,
        description="Title of the retrieved source or webpage.",
    )
    url: str = Field(
        ...,
        description="Canonical URL of the retrieved source.",
    )
    content: str = Field(
        ...,
        description="Relevant snippet, summary, or extracted content from the source.",
    )
    score: float = Field(
        ...,
        description="Relevance score assigned by the search provider.",
    )
    query: str = Field(
        ...,
        description="The exact search query that produced this finding.",
    )


class ReviewResult(BaseModel):
    is_sufficient: bool = Field(
        ...,
        description="Whether the collected sources and findings are sufficient to produce a reliable final answer.",
    )
    feedback: str = Field(
        ...,
        description="If the research is insufficient, explain what is missing or what should be improved.",
    )
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall quality score of the research output between 0.0 and 1.0.",
    )


