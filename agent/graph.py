"""LangGraph assembly: nodes, edges, and the agent loop."""
import json
from langgraph.graph import StateGraph, END
from anthropic import Anthropic
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from .state import AgentState
from .prompts import SYSTEM_PROMPT
from .routing import route_after_reasoning, route_after_checker
from .sanitizer import sanitize_model_text
from tools.definitions import TOOLS
from tools.symbolic_math import run_symbolic_math
from tools.numerical_compute import run_numerical_compute
from tools.plot_results import run_plot
from tools.equations import run_lookup_equation
from config import MODEL, ANTHROPIC_API_KEY, MAX_RETRIES, MAX_TOOL_CALLS

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# --- Map tool names to functions ---
TOOL_DISPATCH = {
    "symbolic_math": run_symbolic_math,
    "numerical_compute": run_numerical_compute,
    "plot_results": run_plot,
    "lookup_equation": run_lookup_equation,
}


# ===== MESSAGE FORMAT CONVERSION =====

def _langchain_to_anthropic(messages):
    """Convert LangGraph messages to Anthropic API message format.

    Handles:
      HumanMessage  -> {"role": "user", "content": "..."}
      AIMessage     -> {"role": "assistant", "content": [text/tool_use blocks]}
      ToolMessage   -> grouped into {"role": "user", "content": [tool_result blocks]}
    """
    anthropic_messages = []
    i = 0

    while i < len(messages):
        msg = messages[i]

        if isinstance(msg, HumanMessage):
            anthropic_messages.append({
                "role": "user",
                "content": msg.content,
            })
            i += 1

        elif isinstance(msg, AIMessage):
            content = []

            # Add text content
            if msg.content:
                if isinstance(msg.content, str) and msg.content.strip():
                    content.append({"type": "text", "text": msg.content})
                elif isinstance(msg.content, list):
                    content.extend(msg.content)

            # Add tool_use blocks from tool_calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["args"],
                    })

            # Anthropic requires non-empty content
            if not content:
                content = [{"type": "text", "text": ""}]

            anthropic_messages.append({
                "role": "assistant",
                "content": content,
            })
            i += 1

        elif isinstance(msg, ToolMessage):
            # Group consecutive ToolMessages into one user message
            # (Anthropic expects all tool_results in a single user turn)
            tool_results = []
            while i < len(messages) and isinstance(messages[i], ToolMessage):
                tm = messages[i]
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tm.tool_call_id,
                    "content": (
                        tm.content
                        if isinstance(tm.content, str)
                        else json.dumps(tm.content)
                    ),
                })
                i += 1

            anthropic_messages.append({
                "role": "user",
                "content": tool_results,
            })

        else:
            # Skip unknown message types
            i += 1

    return anthropic_messages


def _anthropic_to_langchain(response):
    """Convert an Anthropic API response to a LangGraph AIMessage.

    Extracts text blocks and tool_use blocks from the response
    and packages them into a single AIMessage.

    NOTE: No sanitization here — the model's text is preserved as-is
    during intermediate reasoning steps. The expression sourcing checker
    validates the final answer, and the sanitizer is applied only as
    a last-resort fallback after max retries.
    """
    text_parts = []
    tool_calls = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "name": block.name,
                "args": block.input,
            })

    text = "\n".join(text_parts) if text_parts else ""
    return AIMessage(content=text, tool_calls=tool_calls)


# ===== NODE 1: REASONING =====

def reasoning_node(state: AgentState) -> dict:
    """Call the LLM with conversation history and tools.

    The LLM either:
      (a) returns a tool_use block  -> route to tool_execution
      (b) returns text only         -> route to checker
    """
    try:
        # Convert LangGraph messages to Anthropic format
        messages = _langchain_to_anthropic(state["messages"])

        # Call the Anthropic API
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Convert response back to LangGraph message format
        ai_message = _anthropic_to_langchain(response)
        return {"messages": [ai_message]}

    except Exception as e:
        error_msg = f"LLM call failed: {type(e).__name__}: {str(e)}"
        return {
            "messages": [AIMessage(content=error_msg)],
            "stop_reason": "error",
        }


# ===== NODE 2: TOOL EXECUTION =====

def tool_execution_node(state: AgentState) -> dict:
    """Execute the tool the LLM requested.

    1. Extract tool name and arguments from last message
    2. Dispatch to the right function
    3. Return tool result as a new message

    If the tool limit is reached, returns error messages instead of
    executing, so the model knows to present its final answer.
    """
    last_message = state["messages"][-1]

    results = []
    tool_history = list(state["tool_history"])
    figures = list(state["figures"])

    # Safety: if tool limit reached, return errors instead of executing
    if len(tool_history) >= MAX_TOOL_CALLS:
        for tool_call in last_message.tool_calls:
            results.append(ToolMessage(
                content=json.dumps({
                    "error": (
                        f"Tool limit ({MAX_TOOL_CALLS}) reached. "
                        "Present your final answer now using the "
                        "results you have already obtained."
                    ),
                    "success": False,
                }),
                tool_call_id=tool_call["id"],
            ))
        return {
            "messages": results,
            "tool_history": tool_history,
            "figures": figures,
        }

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        # Dispatch to the right tool function
        if tool_name in TOOL_DISPATCH:
            result = TOOL_DISPATCH[tool_name](**tool_args)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        # Track tool usage
        tool_history.append(tool_name)

        # Track generated figures
        if tool_name == "plot_results" and result.get("filepath"):
            figures.append(result["filepath"])

        # Return tool result as a ToolMessage
        results.append(ToolMessage(
            content=json.dumps(result),
            tool_call_id=tool_id,
        ))

    return {
        "messages": results,
        "tool_history": tool_history,
        "figures": figures,
    }


# ===== NODE 3: CHECKER =====

def checker_node(state: AgentState) -> dict:
    """Validate the LLM's answer.

    Runs all applicable checks:
    1. Expression sourcing — model only uses tool-derived expressions
    2. Normalization — wavefunctions are properly normalized
    3. Physical sanity — energy ordering, bound state signs, etc.

    If any fail, increment retry_count and add feedback message.
    On max retries, apply the sanitizer as a fallback and accept.
    """
    from checkers.normalization import extract_and_check_normalizations
    from checkers.physical_sanity import extract_and_check_sanity
    from checkers.expression_sourcing import extract_and_check_expression_sourcing

    checks_passed = True
    feedback = []

    # Check expression sourcing (model only uses tool-derived expressions)
    sourcing_result = extract_and_check_expression_sourcing(state["messages"])
    if not sourcing_result["passed"]:
        checks_passed = False
        feedback.append(sourcing_result["message"])

    # Check normalization of any wavefunctions in tool results
    norm_result = extract_and_check_normalizations(state["messages"])
    if not norm_result["passed"]:
        checks_passed = False
        feedback.append(norm_result["message"])

    # Check physical sanity (energy ordering, bound state signs, etc.)
    sanity_result = extract_and_check_sanity(state["messages"])
    if not sanity_result["passed"]:
        checks_passed = False
        feedback.append(sanity_result["message"])

    if checks_passed:
        return {
            "check_passed": True,
            "stop_reason": "solved",
        }
    else:
        # Check if we've hit max retries
        next_retry = state["retry_count"] + 1
        if next_retry >= MAX_RETRIES and not sourcing_result["passed"]:
            # Last resort: apply sanitizer to the final answer and accept
            last_msg = state["messages"][-1]
            if isinstance(last_msg, AIMessage) and last_msg.content:
                sanitized = sanitize_model_text(last_msg.content)
                return {
                    "check_passed": True,
                    "stop_reason": "solved_sanitized",
                    "messages": [AIMessage(
                        content=sanitized,
                        tool_calls=last_msg.tool_calls if hasattr(last_msg, "tool_calls") else [],
                    )],
                }

        return {
            "check_passed": False,
            "retry_count": next_retry,
            "messages": [HumanMessage(content=(
                "CHECKER FAILED. Issues found:\n"
                + "\n".join(feedback)
                + "\nPlease fix and try again."
            ))],
        }


# ===== BUILD THE GRAPH =====

def build_graph():
    """Assemble the LangGraph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("tool_execution", tool_execution_node)
    graph.add_node("checker", checker_node)

    # Set entry point
    graph.set_entry_point("reasoning")

    # Add conditional edges
    graph.add_conditional_edges(
        "reasoning",
        route_after_reasoning,
        {
            "tool_execution": "tool_execution",
            "checker": "checker",
        }
    )

    # After tool execution, always go back to reasoning
    graph.add_edge("tool_execution", "reasoning")

    graph.add_conditional_edges(
        "checker",
        route_after_checker,
        {
            "reasoning": "reasoning",
            "end": END,
        }
    )

    return graph.compile()


# ===== RUN THE AGENT =====

def run_agent(problem: str) -> AgentState:
    """Run the agent on a physics problem.

    Args:
        problem: Natural language physics question.

    Returns:
        Final AgentState with answer, figures, history.
    """
    app = build_graph()

    initial_state = {
        "messages": [HumanMessage(content=problem)],
        "stop_reason": "",
        "check_passed": False,
        "retry_count": 0,
        "figures": [],
        "tool_history": [],
    }

    final_state = app.invoke(initial_state, config={"recursion_limit": 200})
    return final_state
