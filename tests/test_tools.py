from src.tools.exceptions import ToolInvocationError
from src.tools.search import MockSearchProvider


def test_mock_search_provider_returns_results():
    provider = MockSearchProvider()

    results = provider.search("persian handwritten ocr")

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["title"] == "Mock Search Result"
    assert results[0]["url"] == "https://example.com/mock-result"
    assert "persian handwritten ocr" in results[0]["content"]
    assert results[0]["score"] == 1.0


def test_mock_search_provider_respects_max_results():
    provider = MockSearchProvider()

    results = provider.search("test query", max_results=1)

    assert isinstance(results, list)
    assert len(results) == 1


def test_tool_invocation_error_message():
    error = ToolInvocationError(
        tool_name="web_search",
        message="Something went wrong",
    )

    assert error.tool_name == "web_search"
    assert error.message == "Something went wrong"
    assert str(error) == "Tool:'web_search' invocation failed: Something went wrong"
