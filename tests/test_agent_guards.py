from src.agents.researcher import researcher
from src.agents.reviewer import reviewer


def test_researcher_returns_error_when_no_sub_tasks():
    state = {"current_sub_tasks": []}

    result = researcher(state)

    assert result["errors"] == ["No sub-tasks found to research."]
    assert result["current_node"] == "researcher"


def test_reviewer_forces_publish_when_max_retries_reached():
    state = {
        "count_retries": 2,
        "max_retries": 2,
        "sources": [],
    }

    result = reviewer(state)

    assert result["is_sufficient"] is True
    assert result["current_node"] == "reviewer"


def test_reviewer_forces_publish_when_sources_empty():
    state = {
        "count_retries": 0,
        "max_retries": 2,
        "sources": [],
    }

    result = reviewer(state)

    assert result["is_sufficient"] is True
    assert result["current_node"] == "reviewer"