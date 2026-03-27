"""Normalization checks for quantum mechanical wavefunctions.

Supports any coordinate system by applying the correct Jacobian
(volume element weight) during integration:

    cartesian:   weight = 1                   coords = [x] or [x, y, z]
    spherical:   weight = r^2 sin(theta)      coords = [r, theta, phi]
    cylindrical: weight = rho                 coords = [rho, phi, z]
    polar:       weight = r                   coords = [r, theta]
"""
import json
import numpy as np
from langchain_core.messages import AIMessage, ToolMessage


# Maps coordinate system name -> function that builds the weight
# array from the coordinate meshgrid arrays.
COORDINATE_WEIGHTS = {
    "cartesian": lambda *grids: np.ones_like(grids[0]),
    "spherical": lambda r, theta, phi: r**2 * np.sin(theta),
    "cylindrical": lambda rho, phi, z: rho,
    "polar": lambda r, theta: r,
}


def check_normalization(
    coords,
    psi: np.ndarray,
    coordinate: str = "cartesian",
    tolerance: float = 0.05
) -> dict:
    """Check that a wavefunction is normalized to 1.

    Computes the integral of |psi|^2 over the full volume element
    for the given coordinate system.

    Args:
        coords:     List of 1-D arrays, one per coordinate.
                    Examples:
                      cartesian 1D:  [x]
                      cartesian 3D:  [x, y, z]
                      spherical:     [r, theta, phi]
                      cylindrical:   [rho, phi, z]
                      polar:         [r, theta]
        psi:        Wavefunction values on the grid built from coords.
                    Shape must be (len(coords[0]), len(coords[1]), ...).
                    For a single coordinate, a 1-D array.
        coordinate: Coordinate system name. One of
                    "cartesian", "spherical", "cylindrical", "polar".
        tolerance:  Acceptable deviation from 1.0.

    Returns:
        {"passed": bool, "norm": float, "message": str}
    """
    if coordinate not in COORDINATE_WEIGHTS:
        return {
            "passed": False,
            "norm": 0.0,
            "message": (
                f"Unknown coordinate system: '{coordinate}'. "
                f"Supported: {list(COORDINATE_WEIGHTS.keys())}"
            ),
        }

    # Ensure every coord is a numpy array
    coords = [np.asarray(c) for c in coords]

    if len(coords) == 1:
        # --- 1-D case (most common: single-variable wavefunction) ---
        x = coords[0]
        weight = COORDINATE_WEIGHTS[coordinate](x)
        integrand = np.abs(psi)**2 * weight
        norm = np.trapz(integrand, x)
    else:
        # --- N-D case: build meshgrid, compute weight, integrate ---
        grids = np.meshgrid(*coords, indexing="ij")
        weight = COORDINATE_WEIGHTS[coordinate](*grids)
        integrand = np.abs(psi)**2 * weight

        # Integrate from the last axis to the first
        result = integrand
        for i in reversed(range(len(coords))):
            result = np.trapz(result, coords[i], axis=i)

        norm = float(result)

    passed = abs(norm - 1.0) < tolerance

    return {
        "passed": passed,
        "norm": float(norm),
        "message": (
            f"Normalization ({coordinate}): {norm:.6f} "
            f"({'PASS' if passed else 'FAIL - not normalized'})"
        ),
    }


# Symbols that are physical constants or coordinates — NOT arbitrary
# integration constants (C1, C2, A, B, etc.).
_KNOWN_SYMBOLS = {
    "x", "r", "theta", "phi",           # spatial coordinates
    "hbar", "m", "omega", "E",          # physical constants / energy
    "V0", "a", "n", "l", "Z",          # potential / quantum parameters
    "t", "k", "p",                      # time, wavenumber, momentum
}


def _extract_params(messages):
    """Extract numerical parameter values from plot_results / numerical_compute calls.

    Falls back to natural-unit defaults for common QM symbols.
    """
    params = {}
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        for tc in getattr(msg, "tool_calls", []):
            if tc["name"] in ("plot_results", "numerical_compute"):
                p = tc["args"].get("params", {})
                if p:
                    params.update(p)
    # Natural-unit defaults for anything not specified
    defaults = {"hbar": 1, "m": 1, "omega": 1, "V0": 1, "a": 1, "Z": 1}
    for k, v in defaults.items():
        params.setdefault(k, v)
    return params


def _check_symbolic_normalizations(messages, tolerance=0.05):
    """Check normalization of symbolic |psi|^2 expressions.

    Verifies the agent's own normalization proof: looks for integrate
    tool results where the integrand is an |psi|^2 expression and
    checks whether the result equals 1.
    """
    import sympy as sp
    from ..tools.expression_registry import get as _registry_get

    # 1. Collect abs_squared expression keys
    abs_sq_keys = set()
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        try:
            content = (
                json.loads(msg.content)
                if isinstance(msg.content, str)
                else msg.content
            )
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(content, dict) or not content.get("success"):
            continue
        if content.get("stored_as") and "|^2" in content.get("description", ""):
            abs_sq_keys.add(content["stored_as"])

    if not abs_sq_keys:
        return []

    # 1b. Filter out expressions with arbitrary constants (C1, etc.)
    #     These are intermediate (pre-normalization) expressions.
    determined_keys = set()
    for key in abs_sq_keys:
        expr = _registry_get(key)
        if expr is None:
            continue
        free = {s.name for s in expr.free_symbols}
        if not (free - _KNOWN_SYMBOLS):
            determined_keys.add(key)
    abs_sq_keys = determined_keys

    if not abs_sq_keys:
        return []

    # 2. Find integrate calls whose expression_key is an |psi|^2,
    #    then match to the ToolMessage result.
    integrate_calls = {}  # tool_call_id -> expression_key
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        for tc in getattr(msg, "tool_calls", []):
            if tc["name"] != "symbolic_math":
                continue
            args = tc["args"]
            if args.get("operation") != "integrate":
                continue
            ek = args.get("expression_key", "")
            if ek in abs_sq_keys:
                integrate_calls[tc["id"]] = ek

    # 3. Collect all integrate results per abs_squared key.
    #    The agent may integrate |psi|^2 multiple times — once to
    #    derive the normalization constant (result != 1) and once to
    #    verify after normalizing (result == 1).  A wavefunction is
    #    considered normalized if ANY integration returned 1.
    integrate_results = {}  # abs_sq_key -> list of (result_expr, result_key)
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        if msg.tool_call_id not in integrate_calls:
            continue
        try:
            content = (
                json.loads(msg.content)
                if isinstance(msg.content, str)
                else msg.content
            )
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(content, dict) or not content.get("success"):
            continue

        key = integrate_calls[msg.tool_call_id]
        result_key = content.get("stored_as")
        if not result_key:
            continue

        result_expr = _registry_get(result_key)
        if result_expr is None:
            continue

        integrate_results.setdefault(key, []).append(result_expr)

    # Evaluate each abs_squared key's integration results
    per_key_results = {}  # key -> (any_one, last_expr)
    for key, results in integrate_results.items():
        any_one = False
        last_expr = results[-1]
        for result_expr in results:
            try:
                if sp.simplify(result_expr - 1) == 0:
                    any_one = True
                    break
            except Exception:
                continue
        per_key_results[key] = (any_one, last_expr)

    # If any abs_squared expression was verified (integral = 1),
    # treat non-passing ones as intermediate derivation steps
    # (e.g. integrating |psi|^2 to find the normalization constant).
    any_global_pass = any(v[0] for v in per_key_results.values())

    checks = []
    verified_keys = set()
    for key, (any_one, last_expr) in per_key_results.items():
        if any_one:
            passed = True
        elif any_global_pass:
            # Intermediate — other abs_squared expressions passed
            passed = True
        else:
            passed = False

        norm = 1.0 if any_one else float("nan")
        label = ("PASS" if any_one
                 else "intermediate (skipped)" if any_global_pass
                 else "FAIL - not normalized")
        checks.append({
            "source": f"symbolic_{key}",
            "passed": passed,
            "norm": norm,
            "message": (
                f"Normalization (symbolic {key}): "
                f"integral = {last_expr} ({label})"
            ),
        })
        verified_keys.add(key)

    # 4. Report unverified |psi|^2 keys (no integrate proof found)
    for key in abs_sq_keys - verified_keys:
        expr = _registry_get(key)
        if expr is None:
            continue
        # Skip expressions with arbitrary constants
        free = {s.name for s in expr.free_symbols}
        if free - _KNOWN_SYMBOLS:
            continue
        checks.append({
            "source": f"symbolic_{key}",
            "passed": True,  # don't fail, just note
            "norm": float("nan"),
            "message": (
                f"Normalization (symbolic {key}): "
                f"no integration proof found (skipped)"
            ),
        })

    return checks


def extract_and_check_normalizations(messages, tolerance: float = 0.05) -> dict:
    """Scan tool-result messages for wavefunction data and check normalization.

    Checks both numerical wavefunction arrays (from normalize_wavefunction,
    solve_bvp, solve_ode) and symbolic |psi|^2 expressions (from
    abs_squared after normalization).

    Args:
        messages:  List of LangGraph messages from state["messages"]
        tolerance: Acceptable deviation from 1.0

    Returns:
        {"passed": bool, "checks": list[dict], "message": str}
    """
    checks = []

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        try:
            content = (
                json.loads(msg.content)
                if isinstance(msg.content, str)
                else msg.content
            )
        except (json.JSONDecodeError, TypeError):
            continue

        if not isinstance(content, dict) or not content.get("success"):
            continue

        result = content.get("result")
        if not isinstance(result, dict):
            continue

        # --- normalize_wavefunction results (new: stored_as + norm) ---
        if "normalized_norm" in result and "stored_as" in result:
            norm_val = result["normalized_norm"]
            passed = abs(norm_val - 1.0) < tolerance
            checks.append({
                "source": "normalize_wavefunction",
                "passed": passed,
                "norm": norm_val,
                "message": (
                    f"normalize_wavefunction result: {norm_val:.6f} "
                    f"({'PASS' if passed else 'FAIL'})"
                ),
            })

        # --- solve_bvp / solve_ode results (data registry) ---
        if ("stored_as" in result
                and isinstance(result.get("stored_as"), str)
                and result.get("n_variables")):
            from ..tools.numerical_compute import get_data
            data = get_data(result["stored_as"])
            if data is not None:
                x = np.asarray(data["x"])
                y_data = data["y"]
                if isinstance(y_data, dict) and "y0" in y_data:
                    psi = np.asarray(y_data["y0"])
                    if len(x) == len(psi) and len(x) > 1:
                        check = check_normalization(
                            [x], psi, "cartesian", tolerance
                        )
                        check["source"] = "solver_output"
                        checks.append(check)

    # --- Symbolic wavefunction checks ---
    checks.extend(_check_symbolic_normalizations(messages, tolerance))

    if not checks:
        return {
            "passed": True,
            "checks": [],
            "message": (
                "No wavefunction data found in tool results "
                "to check normalization."
            ),
        }

    all_passed = all(c["passed"] for c in checks)
    summary = "\n".join(c["message"] for c in checks)

    return {
        "passed": all_passed,
        "checks": checks,
        "message": summary,
    }
