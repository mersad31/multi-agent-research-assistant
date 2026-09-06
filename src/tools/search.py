from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
from tavily import TavilyClient

from src.tools.exceptions import ToolInvocationError


class SearchInput(BaseModel):
    """Input schema for the web search tool."""
    query: str = Field(description="The focused search query to look up on the web.")
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of search results to return. Use higher values for deep research."
    )


class SearchProvider(ABC):
    """ Abstract interface for search providers. """

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """ Execute a search query and return normalized search results. """

        raise NotImplementedError


class TavilySearchProvider(SearchProvider):

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ToolInvocationError(
                tool_name="tavily_search",
                message="Missing TAVILY_API_KEY for Tavily search provider.",
            )
        self.client = TavilyClient(api_key=self.api_key)

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:

        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
            )
        except Exception as exc:
            raise ToolInvocationError(
                tool_name="tavily_search",
                message=f"Tavily request failed: {exc}"
            )from exc

        raw_results = response.get("results", [])
        normalized_results: list[dict[str, Any]] = []

        for item in raw_results:
            normalized_results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score"),
                }
            )
        return normalized_results


class MockSearchProvider(SearchProvider):
    """ Mock search provider for local development and testing. """

    def search(
        self,
        query: str,
        max_results: int = 5
    ) -> list[dict[str, Any]]:

        return [
            {
                "title": "Mock Search Result",
                "url": "https://example.com/mock-result",
                "content": (
                    f"This is a mock search result for query: '{query}'. "
                    "Use this only for testing or local development."
                ),
                "score": 1.0,
            }
        ][:max_results]



def create_search_provider(
    use_mock: bool = False,
    api_key: str | None = None,
) -> SearchProvider:
    """ Create and return the appropriate search provider. """

    if use_mock:
        return MockSearchProvider()

    return TavilySearchProvider(api_key=api_key)


def build_search_tool(
        use_mock: bool = False,
        api_key: str | None = None,
) -> BaseTool:
    """ Build a LangChain-compatible search tool. """

    provider = create_search_provider(use_mock=use_mock, api_key=api_key)


    @tool("web_search", args_schema=SearchInput)
    def web_search(query: str, max_results: int = 5) -> str:
        """ Search the web for recent or factual information relevant to a user query. """
        try:
            results = provider.search(query=query, max_results=max_results)
            return json.dumps(results, ensure_ascii=False)

        except ToolInvocationError as e:
            return f"Error executing 'web_search': {e.message}. You may try a different query."

        except Exception as e: # noqa: BLE001 — the tool contract requires returning an error string, never raising
            return f"An unexpected error occurred during search: {e!s}"

    return web_search