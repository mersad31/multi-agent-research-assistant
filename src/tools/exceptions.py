from __future__ import annotations


class ToolInvocationError(Exception):
    """ Raised when a tool invocation fails. """
    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        self.message = message
        super().__init__(message)

    def __str__(self):
        return f"Tool:'{self.tool_name}' invocation failed: {self.message}"
