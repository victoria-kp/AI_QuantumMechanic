"""Expression sourcing checker — ensures the model's final answer
only contains expressions that were returned by tool calls.

The model CAN include mathematical expressions in its answer, but they
must come from tool results (quoted verbatim or as close matches).
Expressions recalled from memory are flagged.

Scalability note
----------------
This checker uses deterministic regex patterns + string matching, which
works for common physics formulas but requires new patterns for each
domain.  A production system would replace the recalled-formula regex
with an LLM-based verification pass: feed the final answer and all tool
expression strings to a lightweight model and ask it to identify claims
not traceable to tool outputs.  This would generalize across domains
without hand-written patterns.
"""

import json
import re
from langchain_core.messages import AIMessage, ToolMessage


# --- Patterns for detecting math notation in text ---

# LaTeX display/inline math
_LATEX_DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_LATEX_INLINE = re.compile(r"\$([^$]+?)\$")

# Unicode math symbols the model might use from memory
_UNICODE_MATH_CHARS = set("ℏψΨφΦωΩ∂∇∫∑∏√∞πθλμνεδΔαβγστκχρ")

# Equation-like lines: contain = with physics-variable content
# (not simple assignments like "n = 0, 1, 2, ...")
_RECALLED_FORMULA = re.compile(
    r"(?:"
    r"E_?\w*\s*=\s*[\(\-a-zA-Z].*?[nhωa-z]{2}"  # E_0, E_n, E_1 = (...) formulas
    r"|[ψΨ]_?\w*\s*\(.*?\)\s*="           # ψ_n(x), ψ_0(x) = ... formulas
    r"|psi_?\w*\s*\(.*?\)\s*="             # psi_n(x), psi_0(x) = ... formulas
    r"|H\s*=\s*-"                           # H = -ℏ²/2m... Hamiltonian defs
    r"|\\frac"                              # LaTeX fractions
    r"|\\partial"                           # LaTeX partial derivatives
    r"|d[²2]\s*[ψfφ]/d[xr][²2]"           # derivative notation
    r")"
)


def _extract_tool_expressions(messages):
    """Collect all expression strings returned by tools.

    Scans ToolMessages for:
    - "expression" field from symbolic_math results
    - "expression" fields from list_stored results

    Returns:
        set of expression strings (normalized: stripped whitespace).
    """
    expressions = set()

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        try:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue

        # Direct expression from symbolic_math
        if "expression" in data and data["expression"]:
            expressions.add(data["expression"].strip())

        # Expressions from list_stored
        if "result" in data and isinstance(data["result"], list):
            for item in data["result"]:
                if isinstance(item, dict) and "expression" in item:
                    expressions.add(item["expression"].strip())

    return expressions


def _normalize(s):
    """Normalize a string for comparison: strip, collapse whitespace."""
    return re.sub(r"\s+", " ", s.strip())


def _has_unicode_math(text):
    """Check if text contains Unicode math symbols."""
    return bool(_UNICODE_MATH_CHARS & set(text))


def _line_contains_tool_expression(line, normalized_tool_exprs):
    """Check if a line contains content derived from tool expressions.

    Uses both exact substring matching and significant overlap detection
    (20+ char chunks) to handle minor reformatting like dropping Eq() wrappers.
    """
    normalized_line = _normalize(line)
    for tool_expr in normalized_tool_exprs:
        # Exact containment (either direction)
        if tool_expr in normalized_line or normalized_line in tool_expr:
            return True
        # Significant overlap: a 20+ char chunk from the line appears
        # in a tool expression (handles partial quoting, Eq() unwrapping, etc.)
        min_chunk = 20
        if len(normalized_line) >= min_chunk:
            for i in range(len(normalized_line) - min_chunk + 1):
                chunk = normalized_line[i:i + min_chunk]
                if chunk in tool_expr:
                    return True
    return False


def _find_non_tool_math(text, tool_expressions):
    """Find mathematical expressions in text that don't match tool results.

    Returns a list of (type, content) tuples for each unmatched expression.
    """
    issues = []
    normalized_tool = {_normalize(e) for e in tool_expressions}

    # Check for LaTeX display math
    for match in _LATEX_DISPLAY.finditer(text):
        issues.append(("LaTeX display math", match.group(0)[:80]))

    # Check for LaTeX inline math
    for match in _LATEX_INLINE.finditer(text):
        issues.append(("LaTeX inline math", match.group(0)[:80]))

    # Check for Unicode math symbols on lines NOT quoting a tool expression
    for line in text.split("\n"):
        if _has_unicode_math(line):
            # If the line contains a tool expression, the Unicode might be
            # part of the quoted expression — allow it
            if not _line_contains_tool_expression(line, normalized_tool):
                issues.append(("Unicode math symbols", line.strip()[:80]))

    # Check for recalled formula patterns not found inside tool expressions
    for match in _RECALLED_FORMULA.finditer(text):
        matched_text = match.group(0)
        # Check if this specific matched text is a substring of any tool expr
        in_tool_expr = any(
            matched_text in tool_expr for tool_expr in tool_expressions
        )
        if not in_tool_expr:
            issues.append(("recalled formula", matched_text[:80]))

    # Check for SymPy-like strings that don't match tool expressions
    for line in text.split("\n"):
        stripped = line.strip()
        # Skip markdown headings, bold text, bullet points
        if stripped.startswith("#") or stripped.startswith("-"):
            continue
        # Look for SymPy-like patterns: foo**2, Derivative(, Eq(, etc.
        if re.search(r"(?:Derivative|Eq|Matrix|sqrt)\s*\(", stripped):
            # This looks like a SymPy expression — check if it's from tools
            if not _line_contains_tool_expression(stripped, normalized_tool):
                issues.append(("novel SymPy expression", stripped[:80]))

    return issues


def extract_and_check_expression_sourcing(messages):
    """Check that the model's final answer only uses tool-derived expressions.

    Follows the standard checker return format:
    {"passed": bool, "issues": list, "message": str}
    """
    # Find the final AI message (the one being checked)
    final_ai = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            final_ai = msg
            break

    if final_ai is None or not final_ai.content:
        return {"passed": True, "issues": [], "message": "No final answer to check."}

    # Collect allowed expressions from tool results
    tool_expressions = _extract_tool_expressions(messages)

    # Find non-tool math in the final answer
    issues = _find_non_tool_math(final_ai.content, tool_expressions)

    if not issues:
        return {
            "passed": True,
            "issues": [],
            "message": "All expressions in the answer are traceable to tool results.",
        }

    # Build feedback message
    issue_lines = []
    for issue_type, content in issues[:5]:  # limit to 5 examples
        issue_lines.append(f"  - {issue_type}: {content}")

    message = (
        "EXPRESSION SOURCING: Your answer contains mathematical "
        "expressions that were not returned by any tool:\n"
        + "\n".join(issue_lines)
        + "\n\nDo NOT write any mathematical expressions that were "
        "not returned by a tool. Only quote expressions from tool "
        "results verbatim. For example:\n"
        '  "The Schrodinger equation (expr_3) is: '
        '(-hbar**2*Derivative(f(x), (x, 2)) + ...)"\n'
        "No LaTeX, no Unicode math symbols, no derivative notation. "
        "Use plain English and reference tool expression keys."
    )

    return {
        "passed": False,
        "issues": issues,
        "message": message,
    }
