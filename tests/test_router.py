import pytest

from src.graph.router import route_after_review


def test_route_after_review_returns_publisher_when_sufficient():
    state = {"is_sufficient": True}

    result = route_after_review(state)

    assert result == "publisher"


def test_route_after_review_returns_replanner_when_insufficient():
    state = {"is_sufficient": False}

    result = route_after_review(state)

    assert result == "replanner"


def test_route_after_review_raises_when_is_sufficient_not_set():
    state = {}

    with pytest.raises(ValueError):
        route_after_review(state)