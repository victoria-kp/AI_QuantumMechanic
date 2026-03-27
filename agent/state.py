"""Agent state definition for LangGraph."""
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """State that flows through every node in the graph.

    Fields:
        messages:      Full conversation history (LLM + tool results).
                       add_messages appends instead of replacing.
        stop_reason:   Why the agent stopped: "solved", "max_retries",
                       "max_tools", "error"
        check_passed:  Did the checker approve the final answer?
        retry_count:   How many times checker sent agent back. Cap at
                       MAX_RETRIES.
        figures:       List of file paths to generated plots.
        tool_history:  List of tool names called, in order.
                       Useful for the presentation.
    """
    messages: Annotated[list, add_messages]
    stop_reason: str
    check_passed: bool
    retry_count: int
    figures: list[str]
    tool_history: list[str]