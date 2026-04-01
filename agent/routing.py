"""Routing functions for LangGraph conditional edges."""
from .state import AgentState
from config import MAX_RETRIES


def route_after_reasoning(state: AgentState) -> str:
    """After the reasoning node, decide: use a tool or
    go to checker?

    Logic: If the last message from the LLM contains a
    tool_use block, route to tool_execution. Otherwise,
    the LLM thinks it's done -> go to checker.
    """
    last_message = state["messages"][-1]

    # Check if LLM wants to use a tool
    # (Anthropic API returns tool_use content blocks)
    # Always route to tool_execution when tool calls exist —
    # tool_execution_node handles the limit by returning error messages.
    # This prevents orphan tool_use blocks with no tool_result.
    if hasattr(last_message, "tool_calls") and \
            last_message.tool_calls:
        return "tool_execution"

    return "checker"


def route_after_checker(state: AgentState) -> str:
    """After the checker node, decide: accept answer or
    retry?

    Logic: If check passed, we're done. If failed and
    retries remain, go back to reasoning. Otherwise,
    stop with best effort.
    """
    if state["check_passed"]:
        return "end"

    if state["retry_count"] < MAX_RETRIES:
        return "reasoning"

    # Max retries hit -- stop with what we have
    return "end"
