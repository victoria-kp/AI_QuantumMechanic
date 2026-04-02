"""Symbolic math tool using SymPy for the AI-QuantumMechanic agent.

All symbolic results are stored in the expression registry as SymPy objects.
The model can SEE expressions in tool results (for reasoning) but must use
expression_key to REFERENCE them in subsequent tool calls.  There is no
raw expression parameter — the model cannot inject guessed expressions.
"""
import re
import sympy as sp
from typing import Literal

# Plain substitution values must be a single number or single symbol.
# Compound expressions (hbar*omega/2, sqrt(2), etc.) must use @key.
_SIMPLE_VALUE = re.compile(
    r'^-?(\d+(\.\d*)?|\.\d+)$'   # numbers: 0, -1, 3.14, .5
    r'|^-?[a-zA-Z_]\w*$'          # single symbol (optionally negated): x, -V0
    r'|^-oo$'                      # negative infinity
)

from .expression_registry import store as _store, get as _registry_get, get_description as _registry_desc, get_label as _registry_label
from .special_functions import ParabolicCylinderD


# ── Session-level assumptions ─────────────────────────────────
_ASSUMPTIONS = {}   # {symbol_name: {assumption: True, ...}}


def _is_assumed_real(name):
    """Check if a symbol is assumed real (explicitly or implied)."""
    assumptions = _ASSUMPTIONS.get(name, {})
    # positive, negative, nonnegative, nonpositive all imply real
    return assumptions.get("real") or assumptions.get("positive") \
        or assumptions.get("negative") or assumptions.get("nonnegative") \
        or assumptions.get("nonpositive")


def _apply_real_assumptions(expr):
    """Strip conjugate wrappers for session-level real symbols.

    Step 1: Replace conjugate(sym) → sym for individual symbols.
    Step 2: For remaining compound conjugates like
            conjugate((m*omega)**(1/4)), strip them if all free
            symbols in the argument are declared real AND the
            argument doesn't contain the imaginary unit I
            (to preserve genuinely complex expressions like
            conjugate(exp(I*m*phi))).
    """
    if not _ASSUMPTIONS:
        return expr
    # Step 1: direct symbol conjugates
    for sym in expr.free_symbols:
        if _is_assumed_real(sym.name):
            expr = expr.subs(sp.conjugate(sym), sym)
    # Step 2: compound conjugates
    if expr.has(sp.conjugate):
        def _strip_if_real(arg):
            if arg.has(sp.I):
                return sp.conjugate(arg)
            if all(_is_assumed_real(s.name) for s in arg.free_symbols):
                return arg
            return sp.conjugate(arg)
        expr = expr.replace(sp.conjugate, _strip_if_real)
    return expr


def _resolve_ref(ref_key):
    """Resolve a key: try registry first, then catalog.

    Returns:
        SymPy object (from registry), string (from catalog), or None.
    """
    resolved = _registry_get(ref_key)
    if resolved is not None:
        return resolved  # SymPy object
    from .equations import resolve_expression
    return resolve_expression(ref_key)  # string or None


# Standard local_dict for sympify — shared across operations
def _make_local_dict():
    """Build the local_dict for sp.sympify with QM symbols.

    Merges hardcoded defaults with session-level assumptions from
    _ASSUMPTIONS so that SymPy operations (dsolve, simplify, etc.)
    see the correct symbol properties.
    """
    local_dict = {
        "hbar": sp.Symbol("hbar", positive=True),
        "m": sp.Symbol("m", positive=True),
        "omega": sp.Symbol("omega", positive=True),
        "E": sp.Symbol("E"),
        "V0": sp.Symbol("V0", positive=True),
        "a": sp.Symbol("a", positive=True),
        "n": sp.Symbol("n"),
        "l": sp.Symbol("l"),
        "Z": sp.Symbol("Z", positive=True),
        "t": sp.Symbol("t", real=True),
        "k": sp.Symbol("k", real=True),
        "p": sp.Symbol("p", real=True),
        "r": sp.Symbol("r", positive=True),
        "theta": sp.Symbol("theta", real=True),
        "phi": sp.Symbol("phi", real=True),
        "pi": sp.pi,
        "exp": sp.exp,
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "I": sp.I,
        "conjugate": sp.conjugate,
        "Matrix": sp.Matrix,
        "oo": sp.oo,
        "inf": sp.oo,
        "Inf": sp.oo,
        "infinity": sp.oo,
        "Infinity": sp.oo,
        "infty": sp.oo,
        # --- QM special functions ---
        "hermite": sp.hermite,
        "assoc_laguerre": sp.assoc_laguerre,
        "Ynm": sp.Ynm,
        "assoc_legendre": sp.assoc_legendre,
        "factorial": sp.factorial,
        "gamma": sp.gamma,
        "Rational": sp.Rational,
        "Abs": sp.Abs,
        "Piecewise": sp.Piecewise,
        "DiracDelta": sp.DiracDelta,
        "Heaviside": sp.Heaviside,
        "KroneckerDelta": sp.KroneckerDelta,
        "ParabolicCylinderD": ParabolicCylinderD,
    }

    # Merge session-level assumptions on top of hardcoded defaults
    for name, assumptions in _ASSUMPTIONS.items():
        if name in local_dict and isinstance(local_dict[name], sp.Symbol):
            base = local_dict[name]
            merged = {k: v for k, v in base.assumptions0.items()
                      if v is not None}
            merged.update(assumptions)
            local_dict[name] = sp.Symbol(name, **merged)
        elif name not in local_dict:
            local_dict[name] = sp.Symbol(name, **assumptions)

    return local_dict


def _to_sympy(value, local_dict):
    """Convert a value to a SymPy object.

    If already a SymPy object, return as-is.
    If a string, parse with sympify.
    """
    if isinstance(value, str):
        return sp.sympify(value, locals=local_dict)
    return value  # already SymPy


def _error(msg):
    """Return a standard error dict."""
    return {
        "stored_as": None,
        "expression": None,
        "description": None,
        "success": False,
        "error": msg,
    }


def _success(result, desc, label=None):
    """Store result in registry and return success dict with expression."""
    key = _store(result, desc, label=label)
    return {
        "stored_as": key,
        "expression": str(result),
        "description": desc,
        "success": True,
        "error": None,
    }


def _classify_asymptotic(term, lim_var, limit_point):
    """Classify asymptotic behavior of a single term.

    Returns 'diverges', 'decays', 'finite', or 'unknown'.

    Tries sp.limit directly first (correct for products like
    x*exp(-x^2)).  Falls back to as_leading_term → limit for
    custom special functions whose limits SymPy can't evaluate.
    """
    lim_val = None

    # 1. Direct limit (handles standard SymPy expressions)
    if limit_point in (sp.oo, -sp.oo):
        try:
            direct = sp.limit(term, lim_var, limit_point)
            if not isinstance(direct, sp.Limit):
                lim_val = direct
        except Exception:
            pass

    # 2. Fallback: leading-term analysis (for special functions)
    if lim_val is None:
        try:
            leading = term.as_leading_term(lim_var)
        except (NotImplementedError, ValueError, sp.PoleError):
            leading = term
        try:
            if limit_point in (sp.oo, -sp.oo):
                lim_val = sp.limit(leading, lim_var, limit_point)
            else:
                lim_val = leading.subs(lim_var, limit_point)
        except Exception:
            return "unknown"

    if (lim_val in (sp.zoo, sp.oo, -sp.oo)
            or (hasattr(lim_val, 'has')
                and (lim_val.has(sp.oo)
                     or lim_val.has(sp.zoo)))):
        return "diverges"
    elif lim_val == 0:
        return "decays"
    elif lim_val.is_finite:
        return "finite"
    return "unknown"


def run_symbolic_math(
    operation: Literal[
        "solve", "integrate", "differentiate",
        "eigenvalues", "solve_ode", "simplify",
        "commutator",
        "time_evolution", "fourier_transform",
        "substitute", "list_stored",
        "arithmetic",
        "limit", "set_equal", "asymptotic",
        "solveset"
    ],
    variable: str = "x",
    **kwargs
) -> dict:
    """Execute a symbolic math operation.

    All inputs come via expression_key (registry/catalog keys).
    Results are stored in the registry and the expression string is
    returned so the model can reason about it.

    Args:
        operation:  Which operation to perform.
        variable:   Variable name(s), e.g. "x" or "r,theta,phi".
        **kwargs:
            expression_key: Registry or catalog key for the input
                            expression.  Required for most operations.
            equation_key:   [substitute] Catalog key for base equation.
            substitutions:  [substitute] {symbol: value_or_@key}.
            operator_b_key: [commutator] Registry key for operator B.
            hamiltonian_key:[time_evolution] Registry key for H.
            energy:         [time_evolution] Energy as simple string.
            (plus other per-operation kwargs)

    Returns:
        dict with: "stored_as", "expression", "description",
                   "success", "error"
    """
    try:
        global _ASSUMPTIONS

        # --- list_stored: return registry contents ---
        if operation == "list_stored":
            from .expression_registry import list_stored
            return {
                "result": list_stored(),
                "success": True,
                "error": None,
            }

        # --- set_assumptions: declare symbol properties for this session ---
        if operation == "set_assumptions":
            PARAM_MAP = {
                "assume_real": "real",
                "assume_positive": "positive",
                "assume_negative": "negative",
                "assume_integer": "integer",
                "assume_nonnegative": "nonnegative",
                "assume_nonpositive": "nonpositive",
                "assume_even": "even",
                "assume_odd": "odd",
            }
            new_assumptions = {}
            any_provided = False
            for param, sympy_key in PARAM_MAP.items():
                names = kwargs.get(param, [])
                if isinstance(names, str):
                    # Handle comma-separated string: "x, a, hbar"
                    names = [n.strip() for n in names.split(",")
                             if n.strip()]
                if names:
                    any_provided = True
                    for name in names:
                        new_assumptions.setdefault(name, {})[sympy_key] = True
            if not any_provided:
                return _error(
                    "At least one assumption list is required "
                    "(assume_real, assume_positive, assume_integer, "
                    "etc.)."
                )
            _ASSUMPTIONS = new_assumptions
            # Build summary for response
            summary = {}
            for name, assumptions in sorted(_ASSUMPTIONS.items()):
                summary[name] = sorted(assumptions.keys())
            return {
                "assumptions": summary,
                "success": True,
                "error": None,
            }

        # --- clear_assumptions: reset session assumptions ---
        if operation == "clear_assumptions":
            _ASSUMPTIONS = {}
            return {
                "assumptions": {},
                "success": True,
                "error": None,
            }

        # --- Build local_dict and parse variables ---
        local_dict = _make_local_dict()

        if "," in variable:
            var_names = [v.strip() for v in variable.split(",")]
            variables = []
            for v_name in var_names:
                if v_name in local_dict:
                    variables.append(local_dict[v_name])
                else:
                    sym = sp.Symbol(v_name)
                    local_dict[v_name] = sym
                    variables.append(sym)
            x = variables[0]
        else:
            if variable in local_dict and isinstance(
                local_dict[variable], sp.Symbol
            ):
                x = local_dict[variable]
            else:
                x = sp.Symbol(variable)
                local_dict[variable] = x
            variables = [x]

        # Helper: get the variable to operate on
        def get_op_variable():
            diff_var = kwargs.get("diff_variable")
            if diff_var:
                if diff_var in local_dict:
                    return local_dict[diff_var]
                return sp.Symbol(diff_var)
            return x

        # --- Resolve expression_key → SymPy object ---
        expression_key = kwargs.pop("expression_key", "")
        expr = None
        if expression_key:
            resolved = _resolve_ref(expression_key)
            if resolved is None:
                return _error(
                    f"Unknown expression_key: '{expression_key}'. "
                    "Use list_stored to see available keys."
                )
            expr = _to_sympy(resolved, local_dict)

        # ── substitute: resolve catalog/registry keys ────────────
        if operation == "substitute":
            from .equations import resolve_expression
            from .catalog_loader import get_catalog

            eq_key = kwargs.get("equation_key", "")
            substitutions = kwargs.get("substitutions") or {}

            if not eq_key and not expression_key:
                return _error(
                    "equation_key or expression_key is required "
                    "for 'substitute'."
                )

            if expression_key:
                base_expr_str = None
                eq_key = expression_key  # for desc
            else:
                base_expr_str = resolve_expression(eq_key)
                if base_expr_str is None:
                    # Fallback: check the expression registry
                    reg_expr = _registry_get(eq_key)
                    if reg_expr is not None:
                        expr = reg_expr
                        if isinstance(expr, sp.Equality):
                            expr = expr.rhs
                        base_expr_str = None  # use expr directly
                    else:
                        return _error(
                            f"Unknown equation_key: '{eq_key}'. "
                            f"Available catalog: "
                            f"{list(get_catalog().keys())}. "
                            f"For registry keys use "
                            f"expression_key instead."
                        )

            if base_expr_str is not None:
                base_expr = sp.sympify(
                    base_expr_str, locals=local_dict
                )
            else:
                base_expr = expr  # from registry via expression_key

            # Apply substitutions: @key resolves from registry/catalog
            sub_names = []
            for sym_name, value in substitutions.items():
                if isinstance(value, str) and value.startswith("@"):
                    ref_key = value[1:]
                    ref = _resolve_ref(ref_key)
                    if ref is None:
                        return _error(
                            f"Substitution references unknown key: "
                            f"'{ref_key}'."
                        )
                    sub_val = _to_sympy(ref, local_dict)
                    sub_names.append(f"{sym_name}=@{ref_key}")
                else:
                    raw = str(value).strip()
                    if not _SIMPLE_VALUE.match(raw):
                        return _error(
                            f"Substitution value '{value}' is a compound "
                            f"expression. Use '@key' to reference a "
                            f"registry expression, or pass a simple "
                            f"value (single number or symbol name)."
                        )
                    sub_val = sp.sympify(raw, locals=local_dict)
                    sub_names.append(f"{sym_name}=<custom>")

                if sym_name in local_dict:
                    target = local_dict[sym_name]
                else:
                    target = sp.Symbol(sym_name)
                new_expr = base_expr.subs(target, sub_val)
                # If subs didn't match and target has assumptions,
                # retry with a plain symbol (handles identity
                # mismatch between assumed and unassumed symbols)
                if new_expr == base_expr and target.assumptions0:
                    plain = sp.Symbol(sym_name)
                    new_expr = base_expr.subs(plain, sub_val)
                base_expr = new_expr

            result = sp.simplify(base_expr)
            desc = (
                f"Assembled: {eq_key} with {', '.join(sub_names)}"
                if sub_names else f"Assembled: {eq_key}"
            )
            return _success(result, desc)

        # ── All other operations need expression_key ──────────────
        if expr is None:
            return _error(
                f"expression_key is required for '{operation}'. "
                "Pass a registry key (e.g. 'expr_1') or a catalog key."
            )

        if operation == "solve":
            solve_var = get_op_variable()
            solutions = sp.solve(expr, solve_var)
            if isinstance(solutions, list) and len(solutions) > 1:
                keys = []
                for i, sol in enumerate(solutions):
                    sol_desc = f"Solution {i+1} for {solve_var}"
                    keys.append(_success(sol, sol_desc))
                return {
                    "solutions": [
                        {"stored_as": k["stored_as"],
                         "expression": k["expression"]}
                        for k in keys
                    ],
                    "description": f"{len(solutions)} solutions for {solve_var}",
                    "success": True,
                    "error": None,
                }
            # Single solution — unwrap; empty — store as-is
            if isinstance(solutions, list) and len(solutions) == 1:
                result = solutions[0]
                desc = f"Solution for {solve_var}"
            else:
                result = solutions
                desc = f"Solutions for {solve_var}"

        elif operation == "solveset":
            solve_var = get_op_variable()
            domain_str = kwargs.get("domain", "Reals")
            domain_map = {
                "Reals": sp.S.Reals,
                "Integers": sp.S.Integers,
                "Complexes": sp.S.Complexes,
            }
            domain = domain_map.get(domain_str, sp.S.Reals)
            result = sp.solveset(expr, solve_var, domain=domain)

            # Replace SymPy dummy variables (_n) with user-friendly n
            n_sym = local_dict["n"]

            def _flatten_imageset(s):
                """Compose nested ImageSets into a single expr."""
                if not isinstance(s, sp.ImageSet):
                    return None
                body = s.lamda.expr
                dummy = s.lamda.variables[0]
                base = s.base_sets[0]
                # Base is Integers/Naturals — leaf
                if isinstance(base, (
                    sp.sets.fancysets.Integers,
                    sp.sets.fancysets.Naturals,
                    sp.sets.fancysets.Naturals0,
                )):
                    return body.subs(dummy, n_sym)
                # Nested ImageSet
                inner = _flatten_imageset(base)
                if inner is not None:
                    return body.subs(dummy, inner)
                # Intersection (e.g. ImageSet ∩ Interval)
                if isinstance(base, sp.sets.sets.Intersection):
                    for arg in base.args:
                        inner = _flatten_imageset(arg)
                        if inner is not None:
                            return body.subs(dummy, inner)
                # Fallback
                return body.subs(dummy, n_sym)

            if isinstance(result, sp.FiniteSet):
                solutions_list = sorted(list(result), key=str)
                if len(solutions_list) > 1:
                    keys = []
                    for i, sol in enumerate(solutions_list):
                        keys.append(_success(
                            sol, f"Solution {i+1} for {solve_var}"
                        ))
                    return {
                        "solutions": [
                            {"stored_as": k["stored_as"],
                             "expression": k["expression"]}
                            for k in keys
                        ],
                        "description": (
                            f"{len(solutions_list)} solutions "
                            f"for {solve_var}"
                        ),
                        "success": True,
                        "error": None,
                    }
                elif len(solutions_list) == 1:
                    return _success(
                        solutions_list[0],
                        f"Solution for {solve_var}",
                    )
                else:
                    return _success(
                        sp.S.EmptySet,
                        f"No solutions for {solve_var}",
                    )

            elif isinstance(result, sp.ImageSet):
                body = _flatten_imageset(result)
                body = sp.simplify(body)
                return _success(
                    body,
                    f"Solution family for {solve_var}: "
                    f"{body} (n integer)",
                )

            elif isinstance(result, sp.Union):
                families = []
                for component in result.args:
                    if isinstance(component, sp.ImageSet):
                        body = _flatten_imageset(component)
                        body = sp.simplify(body)
                        families.append(_success(
                            body,
                            f"Solution family for {solve_var}: "
                            f"{body} (n integer)",
                        ))
                    elif isinstance(component, sp.FiniteSet):
                        for sol in sorted(list(component), key=str):
                            families.append(_success(
                                sol,
                                f"Solution for {solve_var}",
                            ))
                return {
                    "solution_families": [
                        {"stored_as": f["stored_as"],
                         "expression": f["expression"]}
                        for f in families
                    ],
                    "description": (
                        f"{len(families)} solution families "
                        f"for {solve_var}"
                    ),
                    "success": True,
                    "error": None,
                }

            elif isinstance(result, sp.ConditionSet):
                return _success(
                    result,
                    f"Implicit solution for {solve_var}",
                )

            else:
                return _success(
                    result,
                    f"solveset result for {solve_var}",
                )

        elif operation == "integrate":
            int_var = get_op_variable()
            integrand = _apply_real_assumptions(expr)
            lower = kwargs.get("lower_limit")
            upper = kwargs.get("upper_limit")
            if lower is not None and upper is not None:
                lower = sp.sympify(lower, locals=local_dict)
                upper = sp.sympify(upper, locals=local_dict)
                result = sp.integrate(integrand, (int_var, lower, upper))
                desc = f"Definite integral w.r.t. {int_var}"
            else:
                result = sp.integrate(integrand, int_var)
                desc = f"Indefinite integral w.r.t. {int_var}"

        elif operation == "differentiate":
            d_var = get_op_variable()
            order = kwargs.get("order", 1)
            result = sp.diff(expr, d_var, order)
            desc = f"Derivative (order {order}) w.r.t. {d_var}"

        elif operation == "eigenvalues":
            result = expr.eigenvals()
            desc = "Eigenvalues of matrix"

        elif operation == "solve_ode":
            f = sp.Function("f")
            local_dict["f"] = f
            dep_var = get_op_variable()

            ode = expr
            # Unwrap Eq(expr, 0) → expr
            if isinstance(ode, sp.Eq):
                ode = ode.lhs - ode.rhs

            # --- Normalize ODE: divide so highest deriv has coeff 1 ---
            ode_expanded = sp.expand(ode)
            f_func = f(dep_var)

            max_order = 0
            for term in ode_expanded.atoms(sp.Derivative):
                order = term.args[1][1] if len(term.args[1]) > 1 else 1
                if order > max_order:
                    max_order = order

            if max_order > 0:
                highest_deriv = f(dep_var).diff(dep_var, max_order)
                leading_coeff = ode_expanded.coeff(highest_deriv)
                if leading_coeff != 0 and leading_coeff != 1:
                    ode = sp.simplify(ode / leading_coeff)
                    ode_expanded = sp.expand(ode)

            # --- Solve the ODE ---
            hint = kwargs.get("hint", None)
            if hint:
                result = sp.dsolve(ode, f(dep_var), hint=hint)
            else:
                result = sp.dsolve(ode, f(dep_var))

            result_str = str(result)

            # If only series, try closed-form from Weber equation
            if "O(" in result_str:
                try:
                    f_dd = f(dep_var).diff(dep_var, 2)
                    remainder = sp.expand(ode_expanded - f_dd)
                    coeff_f = sp.simplify(remainder / f_func)

                    coeff_expanded = sp.expand(coeff_f)
                    alpha = coeff_expanded.coeff(dep_var, 0)
                    beta = coeff_expanded.coeff(dep_var, 2)
                    linear = coeff_expanded.coeff(dep_var, 1)

                    if beta != 0 and linear == 0:
                        neg_beta = -beta
                        scale = (4 * neg_beta) ** sp.Rational(1, 4)
                        nu = sp.simplify(
                            alpha / (2 * sp.sqrt(neg_beta))
                            - sp.Rational(1, 2)
                        )
                        z = scale * dep_var

                        C1, C2 = sp.symbols('C1 C2')
                        from .special_functions import (
                            ParabolicCylinderD as D,
                        )

                        result = sp.Eq(
                            f_func,
                            C1 * D(nu, z) + C2 * D(-nu - 1, sp.I * z)
                        )
                except Exception:
                    pass  # keep the series solution

            # ── Apply boundary conditions if provided ───────────
            bcs = kwargs.get("boundary_conditions", None)

            if bcs == "normalizable":
                # Open domain: drop divergent terms, extract quantization
                rhs = result.rhs
                terms = sp.Add.make_args(sp.expand(rhs))
                C_syms = {s for s in rhs.free_symbols
                          if s.name.startswith("C")
                          and s.name[1:].isdigit()}

                constants_to_zero = set()
                boundary_analysis = []

                for limit_point in (sp.oo, -sp.oo):
                    for term in terms:
                        behavior = _classify_asymptotic(
                            term, dep_var, limit_point
                        )
                        term_Cs = {c for c in C_syms if term.has(c)}
                        action = "kept"
                        if behavior == "diverges" and term_Cs:
                            constants_to_zero |= term_Cs
                            action = (
                                f"Set "
                                f"{', '.join(str(c) for c in sorted(term_Cs, key=str))}"
                                f" = 0"
                            )
                        boundary_analysis.append({
                            "term": str(term),
                            "at": str(limit_point),
                            "behavior": behavior,
                            "action": action,
                        })

                # Set divergent constants to zero
                cleaned = rhs
                for c in constants_to_zero:
                    cleaned = cleaned.subs(c, 0)
                cleaned = sp.simplify(cleaned)
                result = sp.Eq(result.lhs, cleaned)

                # Extract quantization conditions from special functions
                from .special_functions import ASYMPTOTIC_FUNCTIONS
                quant_conds = []
                for func_cls in ASYMPTOTIC_FUNCTIONS:
                    for atom in cleaned.atoms(func_cls):
                        nu, z_arg = atom.args[0], atom.args[1]
                        for lp in (sp.oo, -sp.oo):
                            quant_conds.extend(
                                func_cls.quantization_condition(
                                    nu, z_arg, dep_var, lp
                                )
                            )

                desc = (f"ODE solution (order {max_order}) in "
                        f"{dep_var} [BCs: normalizable]")
                out = _success(result, desc)
                out["boundary_analysis"] = boundary_analysis
                out["constants_eliminated"] = [
                    str(c) for c in sorted(constants_to_zero, key=str)
                ]
                if quant_conds:
                    qc_out = []
                    for cond in quant_conds:
                        key = _store(
                            cond, f"Quantization condition: {cond}"
                        )
                        qc_out.append({
                            "stored_as": key,
                            "expression": str(cond),
                        })
                    out["quantization_conditions"] = qc_out
                    out["hint"] = (
                        "Quantization conditions found. Use solve on "
                        "each to find energy eigenvalues, then "
                        "substitute specific values to get wavefunctions."
                    )
                return out

            elif isinstance(bcs, list):
                # Finite domain: apply BCs sequentially
                rhs = result.rhs
                boundary_analysis = []
                remaining_conditions = []

                for bc in bcs:
                    point = sp.sympify(
                        str(bc["point"]), locals=local_dict
                    )
                    value = sp.sympify(
                        str(bc.get("value", "0")), locals=local_dict
                    )
                    rhs_at_pt = rhs.subs(dep_var, point)
                    cond_label = (f"f({bc['point']}) = "
                                  f"{bc.get('value', '0')}")

                    # Find integration constants in this equation
                    C_in_eq = sorted(
                        [s for s in rhs_at_pt.free_symbols
                         if s.name.startswith("C")
                         and s.name[1:].isdigit()],
                        key=str,
                    )

                    if not C_in_eq:
                        # No constants left — verify or store
                        check = sp.simplify(rhs_at_pt - value)
                        if check == 0:
                            boundary_analysis.append({
                                "condition": cond_label,
                                "result": "satisfied",
                                "action": "verified",
                            })
                        else:
                            eq = sp.Eq(
                                sp.simplify(rhs_at_pt), value)
                            ck = _store(eq, f"BC: {cond_label}")
                            remaining_conditions.append({
                                "stored_as": ck,
                                "expression": str(eq),
                            })
                            boundary_analysis.append({
                                "condition": cond_label,
                                "result": str(eq),
                                "action": "remaining condition",
                            })
                        continue

                    eq = sp.Eq(rhs_at_pt, value)

                    # Solve for constants algebraically (like
                    # Mathematica DSolve)
                    solved_any = False
                    for c in C_in_eq:
                        sols = sp.solve(eq, c)
                        if len(sols) == 1:
                            candidate = rhs.subs(c, sols[0])
                            if sp.simplify(candidate) == 0:
                                # Substituting would make the
                                # entire solution trivially zero —
                                # return the equation instead
                                ck = _store(
                                    eq, f"BC: {cond_label}")
                                remaining_conditions.append({
                                    "stored_as": ck,
                                    "expression": str(eq),
                                })
                                boundary_analysis.append({
                                    "condition": cond_label,
                                    "result": str(eq),
                                    "action":
                                        "remaining condition",
                                })
                                solved_any = True
                            else:
                                rhs = candidate
                                boundary_analysis.append({
                                    "condition": cond_label,
                                    "result":
                                        f"{c} = {sols[0]}",
                                    "action": "substituted",
                                })
                                solved_any = True
                            break
                    if not solved_any:
                        ck = _store(eq, f"BC: {cond_label}")
                        remaining_conditions.append({
                            "stored_as": ck,
                            "expression": str(eq),
                        })
                        boundary_analysis.append({
                            "condition": cond_label,
                            "result": str(eq),
                            "action": "remaining condition",
                        })

                rhs = sp.simplify(rhs)
                result = sp.Eq(result.lhs, rhs)
                desc = (f"ODE solution (order {max_order}) in "
                        f"{dep_var} [BCs applied]")
                out = _success(result, desc)
                out["boundary_analysis"] = boundary_analysis
                if remaining_conditions:
                    out["remaining_conditions"] = remaining_conditions
                return out

            # ── No BCs provided: return with hint ─────────────
            desc = f"ODE solution (order {max_order}) in {dep_var}"
            out = _success(result, desc)

            result_syms = {s.name for s in result.free_symbols}
            constants = sorted(result_syms & {"C1", "C2", "C3", "C4"})
            if constants:
                out["hint"] = (
                    f"Solution has constants {constants}. "
                    "Apply boundary conditions to determine them: "
                    "for open domains use asymptotic to find divergent "
                    "terms, for finite domains use substitute to enforce "
                    "boundary values (e.g. psi=0 at walls)."
                )
            return out

        elif operation == "simplify":
            result = _apply_real_assumptions(expr)
            result = sp.simplify(result)
            desc = "Simplified expression"

        elif operation == "commutator":
            # Resolve operator B from registry
            b_key = kwargs.get("operator_b_key", "")
            if not b_key:
                return _error(
                    "operator_b_key is required for 'commutator'. "
                    "Pass the registry key of operator B."
                )
            b_ref = _resolve_ref(b_key)
            if b_ref is None:
                return _error(f"Unknown operator_b_key: '{b_key}'.")
            B = _to_sympy(b_ref, local_dict)

            if hasattr(expr, 'shape') and hasattr(B, 'shape'):
                result = expr * B - B * expr
                result = sp.simplify(result)
            else:
                g = sp.Function("g")
                op_var = get_op_variable()

                Bg = B * g(op_var)
                ABg = (
                    (expr * Bg).doit()
                    if hasattr(expr * Bg, 'doit')
                    else expr * Bg
                )

                Ag = expr * g(op_var)
                BAg = (
                    (B * Ag).doit()
                    if hasattr(B * Ag, 'doit')
                    else B * Ag
                )

                result = sp.simplify(sp.expand(ABg - BAg))
            desc = "Commutator result"

        elif operation == "time_evolution":
            t = local_dict["t"]
            hbar = local_dict["hbar"]

            # Energy: accept either energy_key or simple energy string
            energy_key = kwargs.get("energy_key", "")
            energy = kwargs.get("energy")

            if energy_key:
                e_ref = _resolve_ref(energy_key)
                if e_ref is None:
                    return _error(f"Unknown energy_key: '{energy_key}'.")
                E_val = _to_sympy(e_ref, local_dict)
            elif energy is not None:
                E_val = sp.sympify(energy, locals=local_dict)
            else:
                E_val = None

            if E_val is not None:
                phase = sp.exp(-sp.I * E_val * t / hbar)
                result = sp.simplify(expr * phase)
            else:
                # Hamiltonian-based time evolution
                h_key = kwargs.get("hamiltonian_key", "")
                if not h_key:
                    return _error(
                        "time_evolution requires energy, energy_key, "
                        "or hamiltonian_key."
                    )
                h_ref = _resolve_ref(h_key)
                if h_ref is None:
                    return _error(
                        f"Unknown hamiltonian_key: '{h_key}'."
                    )
                H = _to_sympy(h_ref, local_dict)

                if hasattr(H, 'shape'):
                    U = sp.exp(-sp.I * H * t / hbar)
                    result = U * expr
                else:
                    result = (
                        "General time evolution requires "
                        "eigenstate decomposition. "
                        "Provide energy eigenvalue."
                    )
            desc = "Time-evolved state"

        elif operation == "fourier_transform":
            k_name = kwargs.get("transform_variable", "k")
            if k_name in local_dict:
                k = local_dict[k_name]
            else:
                k = sp.Symbol(k_name, real=True)

            int_var = get_op_variable()
            convention = kwargs.get("convention", "physics")

            if convention == "physics":
                integrand = (
                    expr * sp.exp(-sp.I * k * int_var)
                    / sp.sqrt(2 * sp.pi)
                )
            else:
                integrand = (
                    expr * sp.exp(-2 * sp.pi * sp.I * k * int_var)
                )

            result = sp.integrate(integrand, (int_var, -sp.oo, sp.oo))
            result = sp.simplify(result)
            desc = f"Fourier transform ({convention} convention)"

        elif operation == "limit":
            lim_var = get_op_variable()
            point_str = kwargs.get("point", "")
            if not point_str:
                return _error("limit requires 'point' parameter.")
            point = sp.sympify(point_str, locals=local_dict)
            direction = kwargs.get("direction", None)
            if direction:
                result = sp.limit(expr, lim_var, point, dir=direction)
            else:
                result = sp.limit(expr, lim_var, point)
            desc = f"Limit as {lim_var} -> {point_str}"

        elif operation == "asymptotic":
            lim_var = get_op_variable()
            point_str = kwargs.get("point", "")
            if not point_str:
                return _error("asymptotic requires 'point' parameter.")
            point = sp.sympify(point_str, locals=local_dict)

            # Handle Eq(f(x), rhs) — analyze the RHS
            target = expr.rhs if isinstance(expr, sp.Eq) else expr

            # Decompose into additive terms
            terms = sp.Add.make_args(sp.expand(target))
            term_results = []

            for term in terms:
                behavior = _classify_asymptotic(term, lim_var, point)
                term_results.append({
                    "term": str(term),
                    "behavior": behavior,
                })

            return {
                "stored_as": None,
                "expression": str(target),
                "terms": term_results,
                "description": (
                    f"Asymptotic behavior as {lim_var} -> "
                    f"{point_str}"
                ),
                "success": True,
                "error": None,
            }

        elif operation == "set_equal":
            value_str = kwargs.get("value", "0")
            value = sp.sympify(value_str, locals=local_dict)
            result = sp.Eq(expr, value)
            desc = f"Equation: expression = {value_str}"

        # ── arithmetic ────────────────────────────────────────
        elif operation == "arithmetic":
            op = kwargs.get("op", "")
            if not op:
                return _error(
                    "op is required for arithmetic. Options: "
                    "add, subtract, multiply, divide, power, "
                    "conjugate, abs_squared, negate, "
                    "sin, cos, exp, log, sqrt."
                )
            if expr is None:
                return _error(
                    "expression_key is required for arithmetic."
                )

            # Unpack Eq → RHS if needed
            a = expr.rhs if isinstance(expr, sp.Equality) else expr

            # Resolve second operand for binary ops
            BINARY_OPS = {
                "add", "subtract", "multiply", "divide", "power",
            }
            b = None
            second_key = ""
            if op in BINARY_OPS:
                second_key = kwargs.get("second_key", "")
                if not second_key:
                    return _error(
                        f"second_key is required for binary op "
                        f"'{op}'."
                    )
                resolved_b = _resolve_ref(second_key)
                if resolved_b is None:
                    return _error(
                        f"Unknown second_key: '{second_key}'."
                    )
                b = _to_sympy(resolved_b, local_dict)
                if isinstance(b, sp.Equality):
                    b = b.rhs

            # Readable names for descriptions (verbose) and labels (short)
            a_name = _registry_desc(expression_key)
            b_name = _registry_desc(second_key) if second_key else ""
            a_label = _registry_label(expression_key)
            b_label = _registry_label(second_key) if second_key else ""

            # Compute
            if op == "add":
                result = sp.simplify(a + b)
                desc = f"({a_name}) + ({b_name})"
                label = f"({a_label}) + ({b_label})"
            elif op == "subtract":
                result = sp.simplify(a - b)
                desc = f"({a_name}) - ({b_name})"
                label = f"({a_label}) - ({b_label})"
            elif op == "multiply":
                result = sp.simplify(a * b)
                desc = f"({a_name}) * ({b_name})"
                label = f"({a_label}) * ({b_label})"
            elif op == "divide":
                result = sp.simplify(a / b)
                desc = f"({a_name}) / ({b_name})"
                label = f"({a_label}) / ({b_label})"
            elif op == "power":
                result = sp.simplify(a ** b)
                desc = f"({a_name}) ** ({b_name})"
                label = f"({a_label}) ** ({b_label})"
            elif op == "conjugate":
                result = sp.simplify(sp.conjugate(a))
                desc = f"conjugate({a_name})"
                label = f"conjugate({a_label})"
            elif op == "abs_squared":
                raw = sp.conjugate(a) * a
                raw = _apply_real_assumptions(raw)
                result = sp.simplify(raw)
                desc = f"|{a_name}|^2"
                label = f"|{a_label}|^2"
            elif op == "negate":
                result = sp.simplify(-a)
                desc = f"-({a_name})"
                label = f"-({a_label})"
            elif op == "sin":
                result = sp.simplify(sp.sin(a))
                desc = f"sin({a_name})"
                label = f"sin({a_label})"
            elif op == "cos":
                result = sp.simplify(sp.cos(a))
                desc = f"cos({a_name})"
                label = f"cos({a_label})"
            elif op == "tan":
                result = sp.simplify(sp.tan(a))
                desc = f"tan({a_name})"
                label = f"tan({a_label})"
            elif op == "exp":
                result = sp.simplify(sp.exp(a))
                desc = f"exp({a_name})"
                label = f"exp({a_label})"
            elif op == "log":
                result = sp.simplify(sp.log(a))
                desc = f"log({a_name})"
                label = f"log({a_label})"
            elif op == "sqrt":
                result = sp.simplify(sp.sqrt(a))
                desc = f"sqrt({a_name})"
                label = f"sqrt({a_label})"
            else:
                return _error(
                    f"Unknown arithmetic op: '{op}'. Options: "
                    "add, subtract, multiply, divide, power, "
                    "conjugate, abs_squared, negate, "
                    "sin, cos, tan, exp, log, sqrt."
                )

            return _success(result, desc, label=label)

        else:
            return _error(f"Unknown operation: {operation}")

        # ── Store result and return with expression ──────────────
        return _success(result, desc)

    except Exception as e:
        return _error(f"{type(e).__name__}: {str(e)}")
