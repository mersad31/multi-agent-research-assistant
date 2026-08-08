from src.state.graph_state import GraphState


def route_after_review(state: GraphState) -> str:
    is_sufficient = state.get("is_sufficient")

    if is_sufficient is True:
        return "publisher"

    if is_sufficient is False:
        return "researcher"

    raise ValueError(
        "route_after_review called before Reviewer set 'is_sufficient'."
    )