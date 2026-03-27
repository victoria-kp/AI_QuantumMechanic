"""Expression registry — stores SymPy objects for tool chaining.

Every symbolic_math operation stores its result here as a SymPy object
and returns the opaque key + expression string to the model.  The model
can SEE expressions (for reasoning) but must use keys to REFERENCE them
in subsequent tool calls.

Internal helpers (get) return the actual SymPy object for use by tools.
"""

_REGISTRY = {}   # key -> {"expr": sympy.Expr, "description": str, "label": str}
_counter = 0


def store(expr_obj, description, label=None):
    """Store a SymPy object and return an opaque key.

    Args:
        expr_obj:    A SymPy expression/equation object.
        description: Human-readable summary (verbose, for LLM reasoning).
        label:       Short plot-friendly label (e.g. "psi(n=0)").
                     Defaults to description if not provided.

    Returns:
        str: Opaque key like "expr_1".
    """
    global _counter
    _counter += 1
    key = f"expr_{_counter}"
    _REGISTRY[key] = {
        "expr": expr_obj,
        "description": description,
        "label": label or description,
    }
    return key


def get(key):
    """Return the SymPy object for a registry key.

    This is internal — only tools should call it.
    Returns None if key not found.
    """
    entry = _REGISTRY.get(key)
    return entry["expr"] if entry else None


def get_description(key):
    """Return description string for a registry key.

    Returns the key itself if not found.
    """
    entry = _REGISTRY.get(key)
    return entry["description"] if entry else key


def get_label(key):
    """Return the short plot-friendly label for a registry key.

    Returns the key itself if not found.
    """
    entry = _REGISTRY.get(key)
    return entry["label"] if entry else key


def get_str(key):
    """Return string representation of a stored expression.

    Returns None if key not found.
    """
    entry = _REGISTRY.get(key)
    return str(entry["expr"]) if entry else None


def list_stored():
    """Return all stored entries as [{key, description, expression}].

    Includes expression strings so the model can reason about them.
    """
    return [
        {"key": k, "description": v["description"],
         "expression": str(v["expr"])}
        for k, v in _REGISTRY.items()
    ]


def clear():
    """Reset the registry for a new session."""
    global _counter
    _REGISTRY.clear()
    _counter = 0
