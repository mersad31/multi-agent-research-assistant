from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.state.graph_state import GraphState

from src.agents.researcher import researcher
from src.agents.reviewer import reviewer
from src.agents.publisher import publisher
from src.agents.planner import planner

from src.graph.router import route_after_review



def build_graph(checkpointer=None, interrupt_before=None):
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = StateGraph(GraphState)

    # Nodes
    graph.add_node("planner", planner)
    graph.add_node("researcher", researcher)
    graph.add_node("reviewer", reviewer)
    graph.add_node("publisher", publisher)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "researcher": "researcher",
            "publisher": "publisher",
        },
    )

    graph.add_edge("publisher", END)

    compile_kwargs = {"checkpointer": checkpointer}
    if interrupt_before is not None:
        compile_kwargs["interrupt_before"] = interrupt_before

    return graph.compile(**compile_kwargs)
