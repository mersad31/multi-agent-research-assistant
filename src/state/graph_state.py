from enum import Enum
from operator import add
from typing import TypedDict, List, Dict, Optional, Annotated

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
    messages: Annotated[List[AnyMessage], add_messages]
    query: str
    sub_tasks: Annotated[List[Dict], add] #Whole history
    current_sub_tasks: List[Dict] # just current tasks
    sources: Annotated[List[Dict], add]
    findings: Annotated[List[Dict], add] # Reserved for a future Coder/Analyst agent (data analysis output). Not populated yet — use `sources` for raw search results.
    report: str
    current_node: Optional[str]
    max_retries: int
    count_retries: int
    errors: Annotated[List[str], add]
    is_sufficient: Optional[bool]
    review_feedback: Optional[str]