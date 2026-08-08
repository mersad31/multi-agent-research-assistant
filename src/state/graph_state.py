from enum import Enum
from operator import add
from typing import TypedDict, List, Dict, Optional, Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages



class NodeStatus(str, Enum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    REVIEWER = "reviewer"   # This field decide to count retries
    PUBLISHER = "publisher"
    END = "end"


class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    query: str
    sub_tasks: Annotated[List[Dict], add]
    sources: Annotated[List[Dict], add]
    findings: Annotated[List[Dict], add]
    report: str
    current_node: Optional[NodeStatus]
    max_retries: int
    count_retries: int
    errors: Optional[List[str]]   # overwrite: only latest error per node
    is_sufficient: Optional[bool]
    review_feedback: Optional[str]