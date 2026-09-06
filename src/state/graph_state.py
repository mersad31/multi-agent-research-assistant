from enum import Enum
from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class NodeStatus(str, Enum):
    PLANNER = "planner"
    REPLANNER = "replanner"
    RESEARCHER = "researcher"
    REVIEWER = "reviewer"   # This field decide to count retries
    PUBLISHER = "publisher"
    END = "end"


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    sub_tasks: Annotated[list[dict], add] #Whole history
    current_sub_tasks: list[dict] # just current tasks
    sources: Annotated[list[dict], add]
    findings: Annotated[list[dict], add] # Reserved for a future Coder/Analyst agent (data analysis output). Not populated yet — use `sources` for raw search results.
    report: str
    current_node: str | None
    max_retries: int
    count_retries: int
    errors: Annotated[list[str], add]
    is_sufficient: bool | None
    review_feedback: str | None