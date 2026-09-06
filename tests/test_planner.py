from unittest.mock import MagicMock

import src.agents.planner as planner_module
from src.models.schemas import PlanOutput, SubTask


def test_planner_success(monkeypatch):
    fake_output = PlanOutput(
        sub_tasks=[SubTask(description="Research X", search_query="X search query")],
        reasoning="Because X needed one sub-task.",
    )
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = fake_output

    monkeypatch.setattr(planner_module, "_planner_chain", fake_chain)

    result = planner_module.planner({"query": "What is X?"})

    assert result["sub_tasks"] == [
        {"description": "Research X", "search_query": "X search query"}
    ]
    assert result["current_sub_tasks"] == result["sub_tasks"]
    assert result["current_node"] == "planner"
    assert result["errors"] == []


def test_planner_failure(monkeypatch):
    fake_chain = MagicMock()
    fake_chain.invoke.side_effect = Exception("LLM request failed")

    monkeypatch.setattr(planner_module, "_planner_chain", fake_chain)

    result = planner_module.planner({"query": "What is X?"})

    assert result["errors"] == ["Planner node failed: LLM request failed"]
    assert result["current_node"] == "planner"