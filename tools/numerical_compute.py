"""Numerical computation tool using SciPy/NumPy for the AI-QuantumMechanic agent."""
import numpy as np
from scipy import optimize, integrate
from typing import Literal


# ── Data registry — stores numerical arrays, returns opaque keys ─────
_DATA_REGISTRY: dict = {}
_DATA_COUNTER = 0


def _store_data(data: dict) -> str:
    """Store numerical array data and return an opaque key."""
    global _DATA_COUNTER
    _DATA_COUNTER += 1
    key = f"data_{_DATA_COUNTER}"
    _DATA_REGISTRY[key] = data
    return key


def get_data(key: str):
    """Retrieve stored numerical data by key (used by plot_results, checkers)."""
    return _DATA_REGISTRY.get(key)


def _summarize_array(arr):
    """Return [min, max] rounded summary for a numpy array."""
    return [round(float(np.nanmin(arr)), 6), round(float(np.nanmax(arr)), 6)]


def _resolve_expression_key(kwargs, variable="x"):
    """Resolve an expression_key to a NumPy-callable function.

    If expression_key is provided, retrieves the SymPy object from the
    registry, substitutes any params, and converts to a NumPy function
    via lambdify.

    Returns:
        callable or None: A function f(x) -> float, or None if no key.

    Raises:
        ValueError: If the key is not found in the registry.
    """
    expression_key = kwargs.get("expression_key", "")
    if not expression_key:
        return None

    import sympy as sp
    from .expression_registry import get as _registry_get

    expr = _registry_get(expression_key)
    if expr is None:
        raise ValueError(
            f"Unknown expression_key: '{expression_key}'. "
            "Use symbolic_math(list_stored) to see available keys."
        )

    # Substitute known parameter values (match by name, not assumptions)
    params = kwargs.get("params", {})
    sym_map = {s.name: s for s in expr.free_symbols}
    for name, value in params.items():
        if name in sym_map:
            expr = expr.subs(sym_map[name], value)

    # Build lambdify — match variable symbol from expression to preserve assumptions
    var_sym = next((s for s in expr.free_symbols if s.name == variable), sp.Symbol(variable))
    func = sp.lambdify(var_sym, expr, modules=["numpy"])
    return func


def _ode_to_system(kwargs, variable="x", function_name="f"):
    """Convert a registry ODE expression to a numerical first-order system.

    1. Get SymPy expr from registry via expression_key
    2. Handle Eq(lhs, rhs) → lhs - rhs = 0
    3. Find highest derivative order of function_name(variable)
    4. Solve for highest derivative
    5. Substitute f(x)→y0, f'(x)→y1, ... and params
    6. Lambdify into callable rhs(x, y) -> list

    Returns:
        (rhs_callable, order) where rhs_callable(x, y) -> list of floats
        and order is the ODE order (= number of state variables).

    Raises:
        ValueError: If the expression can't be parsed as an ODE.
    """
    import sympy as sp
    from .expression_registry import get as _registry_get

    expression_key = kwargs.get("expression_key", "")
    if not expression_key:
        raise ValueError(
            "expression_key is required. Use symbolic_math to build "
            "the ODE expression, then pass its key."
        )

    expr = _registry_get(expression_key)
    if expr is None:
        raise ValueError(
            f"Unknown expression_key: '{expression_key}'. "
            "Use symbolic_math(list_stored) to see available keys."
        )

    # Handle Eq(lhs, rhs) → lhs - rhs = 0
    if isinstance(expr, sp.Eq):
        ode_expr = expr.lhs - expr.rhs
    else:
        ode_expr = expr

    # Match the variable symbol from the expression (preserves assumptions
    # like real=True set by set_assumptions, so f(x) comparisons work).
    x = next((s for s in ode_expr.free_symbols if s.name == variable), None)
    if x is None:
        # x may only appear inside Derivative args, not as a free symbol
        x = sp.Symbol(variable)
    f = sp.Function(function_name)
    fx = f(x)

    # Find highest derivative order
    max_order = 0
    for term in sp.preorder_traversal(ode_expr):
        if isinstance(term, sp.Derivative):
            # Check if this derivative is of our function
            if term.args[0] == fx:
                order = sum(count for _, count in term.variable_count)
                max_order = max(max_order, order)

    if max_order == 0:
        raise ValueError(
            f"No derivatives of {function_name}({variable}) found in "
            f"expression. Make sure the ODE uses "
            f"Derivative({function_name}({variable}), {variable})."
        )

    # Solve for highest derivative — handle Piecewise by isolating per-branch
    highest_deriv = sp.Derivative(fx, (x, max_order))

    if ode_expr.has(sp.Piecewise):
        # Find the outermost Piecewise, substitute each branch value,
        # then isolate the derivative in each resulting plain expression.
        pw = None
        for arg in sp.preorder_traversal(ode_expr):
            if isinstance(arg, sp.Piecewise):
                pw = arg
                break
        branches = []
        for branch_val, cond in pw.args:
            branch_ode = ode_expr.subs(pw, branch_val)
            sols = sp.solve(branch_ode, highest_deriv)
            if not sols:
                raise ValueError(
                    f"Cannot isolate {highest_deriv} from Piecewise "
                    f"branch where {pw} = {branch_val}."
                )
            branches.append((sols[0], cond))
        rhs_expr = sp.Piecewise(*branches)
    else:
        solutions = sp.solve(ode_expr, highest_deriv)
        if not solutions:
            raise ValueError(
                f"Cannot isolate {highest_deriv} from the ODE. "
                f"The equation may not be in a solvable form."
            )
        rhs_expr = solutions[0]

    # Build substitution: f(x) → y0, f'(x) → y1, f''(x) → y2, ...
    y_syms = [sp.Symbol(f"y{i}") for i in range(max_order)]
    subs = {fx: y_syms[0]}
    for i in range(1, max_order):
        subs[sp.Derivative(fx, (x, i))] = y_syms[i]
    rhs_sub = rhs_expr.subs(subs)

    # Substitute parameter values
    params = kwargs.get("params", {})
    sym_map = {s.name: s for s in rhs_sub.free_symbols}
    for name, value in params.items():
        if name in sym_map:
            rhs_sub = rhs_sub.subs(sym_map[name], value)

    # Build system: dy/dx = [y1, y2, ..., rhs(x, y0, y1, ...)]
    system_exprs = list(y_syms[1:]) + [rhs_sub]
    all_vars = [x] + y_syms
    system_funcs = [
        sp.lambdify(all_vars, expr_i, modules=["numpy"])
        for expr_i in system_exprs
    ]

    def rhs_func(x_val, y_arr):
        return [f_i(x_val, *y_arr) for f_i in system_funcs]

    return rhs_func, max_order


def run_numerical_compute(
    operation: Literal[
        "root_finding", "integrate_quad",
        "find_all_roots",
        "normalize_wavefunction",
        "solve_ode", "solve_bvp",
        "evaluate_grid"
    ],
    **kwargs
) -> dict:
    """Execute a numerical computation.

    All operations that accept expressions require expression_key
    (a registry key from symbolic_math). No raw strings are accepted.

    Args:
        operation: Which numerical operation to perform.
        **kwargs: Operation-specific arguments:

            root_finding:
                expression_key (str): Registry key for the function.
                x0 (float): Initial guess (for fsolve)
                bracket (list[float]): [a, b] bracket (for brentq)
                params (dict): Parameter values for expression_key

            find_all_roots:
                expression_key (str): Registry key for the function.
                x_min (float): Lower search bound
                x_max (float): Upper search bound
                n_points (int): Number of grid points for sign changes
                params (dict): Parameter values for expression_key

            integrate_quad:
                expression_key (str): Registry key for the integrand.
                lower (float): Lower bound
                upper (float): Upper bound (use 1e308 for infinity)
                params (dict): Parameter values for expression_key

            solve_ode:
                expression_key (str): Registry key for the ODE (Eq or
                    expression=0). The tool auto-decomposes it into a
                    first-order system.
                variable (str): Independent variable (default "x")
                function_name (str): Dependent function (default "f")
                x_span (list[float]): [x_start, x_end]
                y0 (list[float]): Initial conditions [y(0), y'(0), ...]
                n_points (int): Number of output points (default 500)
                params (dict): Named parameters, e.g. {"omega": 2.0}
                method (str): Integration method (default "RK45")

            solve_bvp:
                expression_key (str): Registry key for the ODE.
                boundary_conditions (list[dict]): Structured BCs.
                    Each: {side: "left"|"right", variable_index: int,
                    value: float}
                variable (str): Independent variable (default "x")
                function_name (str): Dependent function (default "f")
                x_span (list[float]): [x_start, x_end]
                y_guess (list[list[float]]): Initial guess (n_vars, n_points)
                n_points (int): Number of mesh points (default 100)
                params (dict): Named parameters
                expected_nodes (int): Expected zero crossings

            normalize_wavefunction:
                x_data (list[float]): Position array
                psi_data (list[float]): Wavefunction values

    Returns:
        dict with keys: "result" (JSON-serializable),
                        "success" (bool), "error" (str or None)
    """
    try:
        if operation == "root_finding":
            func = _resolve_expression_key(kwargs)
            if func is None:
                return {
                    "result": None, "success": False,
                    "error": (
                        "expression_key is required. Use symbolic_math "
                        "to build the expression, then pass its key."
                    )
                }

            if "bracket" in kwargs:
                a, b = kwargs["bracket"]
                root = optimize.brentq(func, a, b)
            else:
                x0 = kwargs.get("x0", 1.0)
                root = optimize.fsolve(func, x0)[0]
                if abs(func(root)) > 1e-8:
                    return {
                        "result": None, "success": False,
                        "error": f"fsolve did not converge. f(root) = {func(root):.2e}"
                    }

            result = {
                "root": float(root),
                "verification": float(func(root))
            }

        elif operation == "find_all_roots":
            x_min = kwargs.get("x_min", 0.0)
            x_max = kwargs.get("x_max", 10.0)
            n_points = kwargs.get("n_points", 1000)

            func = _resolve_expression_key(kwargs)
            if func is None:
                return {
                    "result": None, "success": False,
                    "error": (
                        "expression_key is required. Use symbolic_math "
                        "to build the expression, then pass its key."
                    )
                }

            x_grid = np.linspace(x_min, x_max, n_points)

            y_grid = np.zeros_like(x_grid)
            valid = np.ones(len(x_grid), dtype=bool)
            for i, xi in enumerate(x_grid):
                try:
                    val = func(xi)
                    if np.isfinite(val):
                        y_grid[i] = val
                    else:
                        valid[i] = False
                except Exception:
                    valid[i] = False

            roots = []
            valid_indices = np.where(valid)[0]
            for i in range(len(valid_indices) - 1):
                idx1 = valid_indices[i]
                idx2 = valid_indices[i + 1]
                if idx2 - idx1 == 1 and y_grid[idx1] * y_grid[idx2] < 0:
                    try:
                        root = optimize.brentq(func, x_grid[idx1], x_grid[idx2])
                        if abs(func(root)) < 1e-8:
                            roots.append(float(root))
                    except Exception:
                        pass

            result = {
                "roots": roots,
                "n_roots": len(roots),
                "search_interval": [float(x_min), float(x_max)]
            }

        elif operation == "integrate_quad":
            a = kwargs["lower"]
            b = kwargs["upper"]
            # Convert large bounds to np.inf (scipy.quad supports it)
            if a <= -1e300:
                a = -np.inf
            if b >= 1e300:
                b = np.inf

            func = _resolve_expression_key(kwargs)
            if func is None:
                return {
                    "result": None, "success": False,
                    "error": (
                        "expression_key is required. Use symbolic_math "
                        "to build the expression, then pass its key."
                    )
                }

            value, error = integrate.quad(func, a, b)
            result = {
                "value": float(value),
                "absolute_error": float(error)
            }

        elif operation == "solve_ode":
            variable = kwargs.get("variable", "x")
            function_name = kwargs.get("function_name", "f")
            x_span = kwargs["x_span"]
            y0 = kwargs["y0"]
            n_points = kwargs.get("n_points", 500)
            method = kwargs.get("method", "RK45")

            rhs, order = _ode_to_system(kwargs, variable, function_name)

            if len(y0) != order:
                return {
                    "result": None, "success": False,
                    "error": (
                        f"ODE is order {order}, so y0 must have "
                        f"{order} elements [y(0), y'(0), ...], "
                        f"but got {len(y0)} elements."
                    )
                }

            x_eval = np.linspace(x_span[0], x_span[1], n_points)

            sol = integrate.solve_ivp(
                rhs, x_span, y0,
                t_eval=x_eval,
                method=method,
                rtol=1e-10,
                atol=1e-12
            )

            if not sol.success:
                return {
                    "result": None, "success": False,
                    "error": (
                        f"solve_ivp failed: {sol.message}. "
                        f"Tips: (1) Try a smaller x_span. "
                        f"(2) For stiff problems use method='Radau' or 'BDF'. "
                        f"(3) Check your equations and initial conditions."
                    )
                }

            n_vars = len(y0)

            # Store arrays in data registry (not sent to API)
            data = {
                "x": sol.t,
                "y": {f"y{i}": sol.y[i] for i in range(n_vars)},
            }
            data_key = _store_data(data)

            summary = {
                "n_points": len(sol.t),
                "x_range": _summarize_array(sol.t),
            }
            for i in range(n_vars):
                summary[f"y{i}_range"] = _summarize_array(sol.y[i])

            result = {
                "stored_as": data_key,
                "summary": summary,
                "n_variables": n_vars,
                "method": method,
            }

            # --- Validate IVP solution ---
            warnings = []

            for i in range(n_vars):
                y_arr = np.array(sol.y[i])
                if np.any(~np.isfinite(y_arr)):
                    warnings.append(
                        f"BLOWUP: Variable y{i} contains inf/nan. "
                        f"Try a smaller x_span, different method (e.g. 'Radau' "
                        f"for stiff problems), or check your equations."
                    )
                elif np.max(np.abs(y_arr)) > 1e15:
                    warnings.append(
                        f"LARGE VALUES: Variable y{i} reaches {np.max(np.abs(y_arr)):.2e}. "
                        f"Solution may be unstable. Check parameters or try "
                        f"a different integration method."
                    )

            if warnings:
                result["warnings"] = warnings

        elif operation == "solve_bvp":
            variable = kwargs.get("variable", "x")
            function_name = kwargs.get("function_name", "f")
            x_span = kwargs["x_span"]
            n_points = kwargs.get("n_points", 100)
            boundary_conditions = kwargs.get("boundary_conditions", [])

            if not boundary_conditions:
                return {
                    "result": None, "success": False,
                    "error": (
                        "boundary_conditions is required. Provide a list "
                        "of {side, variable_index, value} dicts. "
                        "Example: [{\"side\":\"left\",\"variable_index\":0,"
                        "\"value\":0},{\"side\":\"right\","
                        "\"variable_index\":0,\"value\":0}]"
                    )
                }

            rhs_scalar, order = _ode_to_system(
                kwargs, variable, function_name
            )

            # Vectorized RHS for solve_bvp (handles x as array)
            def rhs(x, y):
                result_arr = np.zeros_like(y)
                for i in range(x.shape[0] if hasattr(x, 'shape') else 1):
                    xi = float(x[i]) if hasattr(x, '__getitem__') else float(x)
                    yi = y[:, i] if y.ndim > 1 else y
                    vals = rhs_scalar(xi, yi)
                    for j, v in enumerate(vals):
                        result_arr[j, i] = v
                return result_arr

            # Build bc function from structured boundary_conditions
            def bc(ya, yb):
                residuals = []
                for cond in boundary_conditions:
                    arr = ya if cond["side"] == "left" else yb
                    idx = cond.get("variable_index", 0)
                    val = cond.get("value", 0.0)
                    residuals.append(arr[idx] - val)
                return np.array(residuals)

            x_mesh = np.linspace(x_span[0], x_span[1], n_points)

            if "y_guess" in kwargs:
                y_guess = np.array(kwargs["y_guess"])
            else:
                y_guess = np.zeros((order, n_points))
                y_guess[0] = np.sin(
                    np.pi * (x_mesh - x_span[0])
                    / (x_span[1] - x_span[0])
                )
                if order > 1:
                    y_guess[1] = np.gradient(y_guess[0], x_mesh)

            sol = integrate.solve_bvp(rhs, bc, x_mesh, y_guess)

            if not sol.success:
                return {
                    "result": None, "success": False,
                    "error": (
                        f"solve_bvp failed: {sol.message}. "
                        f"Try a better initial guess. Tips: "
                        f"(1) Use a sine or Gaussian shape for bound states. "
                        f"(2) Make sure y_guess shape is (n_vars, n_points). "
                        f"(3) Include derivative guess as second row. "
                        f"(4) Check that boundary conditions are correct."
                    )
                }

            x_fine = np.linspace(x_span[0], x_span[1], 500)
            y_fine = sol.sol(x_fine)

            # --- Validate BVP solution ---
            warnings = []

            # Check 1: Trivial solution?
            max_amplitude = np.max(np.abs(y_fine[0]))
            if max_amplitude < 1e-10:
                warnings.append(
                    "TRIVIAL SOLUTION: The solver found psi ~ 0 everywhere. "
                    "This satisfies the boundary conditions but is not physical. "
                    "Try a different initial guess with larger amplitude, "
                    "or adjust the eigenvalue parameter."
                )

            # Check 2: Residual too large?
            max_residual = float(np.max(np.abs(sol.rms_residuals)))
            if max_residual > 1e-4:
                warnings.append(
                    f"HIGH RESIDUAL: {max_residual:.2e}. Solution may be inaccurate. "
                    f"Try more mesh points or a better initial guess."
                )

            # Check 3: Blowup at boundaries?
            boundary_ratio = max(
                np.abs(y_fine[0, 0]), np.abs(y_fine[0, -1])
            ) / (max_amplitude + 1e-30)
            if boundary_ratio > 0.1 and max_amplitude > 1e-10:
                warnings.append(
                    f"BOUNDARY BLOWUP: Solution has significant amplitude at "
                    f"boundaries ({boundary_ratio:.2%} of max). For bound states, "
                    f"psi should decay to 0. The energy eigenvalue may be wrong."
                )

            # Check 4: Node count
            zero_crossings = np.sum(np.diff(np.sign(y_fine[0])) != 0)
            n_expected = kwargs.get("expected_nodes", None)
            if n_expected is not None and zero_crossings != n_expected:
                warnings.append(
                    f"NODE COUNT: Found {zero_crossings} nodes but expected "
                    f"{n_expected}. This may be a higher/lower excited state "
                    f"than intended. Adjust the energy parameter."
                )

            n_vars = y_fine.shape[0]

            # Store arrays in data registry
            data = {
                "x": x_fine,
                "y": {f"y{i}": y_fine[i] for i in range(n_vars)},
            }
            data_key = _store_data(data)

            summary = {
                "n_points": len(x_fine),
                "x_range": _summarize_array(x_fine),
            }
            for i in range(n_vars):
                summary[f"y{i}_range"] = _summarize_array(y_fine[i])

            result = {
                "stored_as": data_key,
                "summary": summary,
                "n_variables": n_vars,
                "residual_norm": max_residual,
                "max_amplitude": float(max_amplitude),
                "zero_crossings": int(zero_crossings),
            }

            if warnings:
                result["warnings"] = warnings

        elif operation == "normalize_wavefunction":
            data_key = kwargs.get("data_key", "")
            variable_index = kwargs.get("variable_index", 0)

            if data_key:
                data = get_data(data_key)
                if data is None:
                    return {
                        "result": None, "success": False,
                        "error": (
                            f"Unknown data_key: '{data_key}'. "
                            "Use a key returned by solve_ode or solve_bvp."
                        ),
                    }
                x = np.asarray(data["x"])
                y_data = data["y"]
                if isinstance(y_data, dict):
                    var_key = f"y{variable_index}"
                    if var_key not in y_data:
                        return {
                            "result": None, "success": False,
                            "error": (
                                f"variable_index {variable_index} not in data. "
                                f"Available: {list(y_data.keys())}"
                            ),
                        }
                    psi = np.asarray(y_data[var_key])
                else:
                    psi = np.asarray(y_data)
            elif "x_data" in kwargs and "psi_data" in kwargs:
                x = np.array(kwargs["x_data"])
                psi = np.array(kwargs["psi_data"])
            else:
                return {
                    "result": None, "success": False,
                    "error": (
                        "data_key is required. Pass a key from "
                        "solve_ode or solve_bvp."
                    ),
                }

            norm_sq = np.trapz(np.abs(psi)**2, x)
            norm = np.sqrt(norm_sq)

            if norm < 1e-15:
                return {
                    "result": None, "success": False,
                    "error": "Wavefunction is essentially zero everywhere"
                }

            psi_normalized = psi / norm
            norm_check = np.trapz(np.abs(psi_normalized)**2, x)

            # Store normalized result in data registry
            norm_data = {"x": x, "y": psi_normalized}
            norm_key = _store_data(norm_data)

            result = {
                "stored_as": norm_key,
                "original_norm": float(norm),
                "normalized_norm": float(norm_check),
                "summary": {
                    "n_points": len(x),
                    "x_range": _summarize_array(x),
                    "y_range": _summarize_array(psi_normalized),
                },
            }

        elif operation == "evaluate_grid":
            x_min = kwargs.get("x_min", -5.0)
            x_max = kwargs.get("x_max", 5.0)
            n_points = kwargs.get("n_points", 500)
            variable = kwargs.get("variable", "x")

            # Support single key or list of keys
            expression_keys = kwargs.get("expression_keys", [])
            single_key = kwargs.get("expression_key", "")

            if single_key and not expression_keys:
                expression_keys = [single_key]

            if not expression_keys:
                return {
                    "result": None, "success": False,
                    "error": (
                        "expression_key or expression_keys is "
                        "required for evaluate_grid."
                    )
                }

            x_grid = np.linspace(x_min, x_max, n_points)

            datasets = []
            for key in expression_keys:
                func = _resolve_expression_key(
                    {"expression_key": key,
                     "params": kwargs.get("params", {})},
                    variable=variable
                )
                if func is None:
                    return {
                        "result": None, "success": False,
                        "error": f"Could not resolve key: '{key}'."
                    }

                y_grid = np.empty_like(x_grid)
                for i, xi in enumerate(x_grid):
                    try:
                        val = float(func(xi))
                        y_grid[i] = val if np.isfinite(val) else 0.0
                    except Exception:
                        y_grid[i] = 0.0

                datasets.append({
                    "key": key,
                    "y": y_grid,
                })

            if len(datasets) == 1:
                dk = _store_data({"x": x_grid, "y": datasets[0]["y"]})
                result = {
                    "stored_as": dk,
                    "summary": {
                        "n_points": len(x_grid),
                        "x_range": _summarize_array(x_grid),
                        "y_range": _summarize_array(datasets[0]["y"]),
                    },
                }
            else:
                data_keys = []
                ds_summaries = []
                for ds in datasets:
                    dk = _store_data({"x": x_grid, "y": ds["y"]})
                    data_keys.append(dk)
                    ds_summaries.append({
                        "data_key": dk,
                        "source_expression": ds["key"],
                        "y_range": _summarize_array(ds["y"]),
                    })
                result = {
                    "stored_as": data_keys,
                    "summary": {
                        "n_points": len(x_grid),
                        "x_range": _summarize_array(x_grid),
                        "datasets": ds_summaries,
                    },
                }

        else:
            return {
                "result": None, "success": False,
                "error": f"Unknown operation: {operation}"
            }

        return {"result": result, "success": True, "error": None}

    except Exception as e:
        return {
            "result": None, "success": False,
            "error": f"{type(e).__name__}: {str(e)}"
        }