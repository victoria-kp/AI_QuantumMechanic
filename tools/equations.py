"""Equation catalog and lookup tool for the AI-QuantumMechanic agent.

Provides a structured catalog of famous physics / quantum mechanics
equations as SymPy-parseable strings.  The model calls this tool to
retrieve equation *keys* (not raw expressions).  It must then pass
those keys to symbolic_math(substitute, ...) to assemble and use
the actual expressions — the model never sees the formula text.

The catalog is loaded from JSON files in data/equations/ via
catalog_loader.  Semantic search is powered by ChromaDB via
vector_store.
"""

from tools.catalog_loader import get_catalog
from tools.vector_store import search_equations


def _get_catalog():
    """Return the equation catalog (loaded from JSON on first call)."""
    return get_catalog()


# ── Internal helper (used by symbolic_math) ─────────────────────

def resolve_expression(key: str):
    """Return the raw SymPy expression string for a catalog key.

    This is an *internal* function — it is NOT exposed to the model.
    Only ``symbolic_math.substitute`` should call it.
    """
    entry = _get_catalog().get(key)
    return entry["expression"] if entry else None


def _find_key(name: str):
    """Exact-or-fuzzy match a name to a catalog key."""
    catalog = _get_catalog()
    if name in catalog:
        return name
    name_lower = name.lower().replace(" ", "_").replace("-", "_")
    matches = [
        key for key, entry in catalog.items()
        if name_lower in key
        or name_lower in entry["name"].lower().replace(" ", "_")
    ]
    return matches[0] if len(matches) == 1 else None


# ── Tool handler ─────────────────────────────────────────────────

def run_lookup_equation(
    operation: str,
    name: str = "",
    tag: str = "",
    query: str = "",
    n_results: int = 5,
) -> dict:
    """Look up equations from the catalog.

    The ``get`` operation returns metadata (name, description,
    variables, symbols, tags) but **not** the raw expression.
    To use the expression, pass its key to
    ``symbolic_math(operation="substitute", ...)``.

    Args:
        operation: "list" to see available equations, "get" to
                   retrieve metadata for one, "search" for
                   semantic search.
        name:      [get] Canonical key of the equation.
        tag:       [list/search] Optional tag to filter results.
        query:     [search] Natural-language search query.
        n_results: [search] Max results to return (default 5).

    Returns:
        dict with keys: "result", "success", "error"
    """
    catalog = _get_catalog()

    try:
        if operation == "list":
            entries = []
            for key, entry in catalog.items():
                if tag and tag.lower() not in [t.lower() for t in entry["tags"]]:
                    continue
                entries.append({
                    "key": key,
                    "name": entry["name"],
                    "description": entry["description"],
                    "tags": entry["tags"],
                })
            return {"result": entries, "success": True, "error": None}

        elif operation == "get":
            if not name:
                return {
                    "result": None,
                    "success": False,
                    "error": "Parameter 'name' is required for operation 'get'.",
                }

            resolved_key = _find_key(name)

            if resolved_key:
                entry = catalog[resolved_key]
                # Return metadata WITHOUT the expression
                return {
                    "result": {
                        "key": resolved_key,
                        "name": entry["name"],
                        "description": entry["description"],
                        "variables": entry["variables"],
                        "symbols_used": entry["symbols_used"],
                        "tags": entry["tags"],
                        "usage": (
                            f"Pass equation_key=\"{resolved_key}\" to "
                            "symbolic_math(operation=\"substitute\", ...) "
                            "to assemble this equation."
                        ),
                    },
                    "success": True,
                    "error": None,
                }

            # No match — check for ambiguous
            name_lower = name.lower().replace(" ", "_").replace("-", "_")
            matches = [
                key for key, entry in catalog.items()
                if name_lower in key
                or name_lower in entry["name"].lower().replace(" ", "_")
            ]
            if len(matches) > 1:
                return {
                    "result": None,
                    "success": False,
                    "error": (
                        f"Ambiguous name '{name}'. Multiple matches: "
                        f"{matches}. Use the exact key."
                    ),
                }
            else:
                available = list(catalog.keys())
                return {
                    "result": None,
                    "success": False,
                    "error": (
                        f"Equation '{name}' not found. "
                        f"Available keys: {available}"
                    ),
                }

        elif operation == "search":
            if not query:
                return {
                    "result": None,
                    "success": False,
                    "error": "Parameter 'query' is required for operation 'search'.",
                }
            hits = search_equations(query=query, n_results=n_results, tag=tag)
            return {"result": hits, "success": True, "error": None}

        else:
            return {
                "result": None,
                "success": False,
                "error": (
                    f"Unknown operation: {operation}. "
                    "Use 'list', 'get', or 'search'."
                ),
            }

    except Exception as e:
        return {
            "result": None,
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
        }
