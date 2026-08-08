import operator
from typing import get_args, get_origin

from src.state.graph_state import GraphState, NodeStatus


def test_graph_state_has_expected_fields():
    annotations = GraphState.__annotations__

    expected_fields = {
        "messages",
        "query",
        "sub_tasks",
        "sources",
        "findings",
        "report",
        "current_node",
        "max_retries",
        "count_retries",
        "errors",
        "is_sufficient",
        "review_feedback",
    }

    assert expected_fields == set(annotations.keys())


def test_graph_state_uses_add_for_sub_tasks():
    sub_tasks_annotation = GraphState.__annotations__["sub_tasks"]

    assert get_origin(sub_tasks_annotation) is not None
    assert operator.add in get_args(sub_tasks_annotation)


def test_graph_state_uses_add_messages_for_messages():
    messages_annotation = GraphState.__annotations__["messages"]

    assert get_origin(messages_annotation) is not None
    metadata = get_args(messages_annotation)
    assert len(metadata) > 1


def test_operator_add_combines_lists():
    messages_a = [{"role": "user", "content": "hello"}]
    messages_b = [{"role": "assistant", "content": "hi"}]
    sub_tasks_a = [{"search_query": "ocr"}]
    sub_tasks_b = [{"search_query": "clustering"}]

    assert operator.add(messages_a, messages_b) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    assert operator.add(sub_tasks_a, sub_tasks_b) == [
        {"search_query": "ocr"},
        {"search_query": "clustering"},
    ]


def test_node_status_values():
    assert NodeStatus.PLANNER.value == "planner"
    assert NodeStatus.RESEARCHER.value == "researcher"
    assert NodeStatus.REVIEWER.value == "reviewer"
    assert NodeStatus.PUBLISHER.value == "publisher"
    assert NodeStatus.END.value == "end"
