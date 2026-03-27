"""Equation catalog and lookup tool for the AI-QuantumMechanic agent.

Provides a structured catalog of famous physics / quantum mechanics
equations as SymPy-parseable strings.  The model calls this tool to
retrieve equation *keys* (not raw expressions).  It must then pass
those keys to symbolic_math(substitute, ...) to assemble and use
the actual expressions — the model never sees the formula text.
"""


EQUATION_CATALOG = {
    # ── Schrodinger equations ────────────────────────────────────
    "time_independent_schrodinger_1d": {
        "name": "Time-Independent Schrodinger Equation (1D)",
        "expression": "-hbar**2/(2*m) * f(x).diff(x, 2) + V*f(x) - E*f(x)",
        "description": (
            "H psi = E psi in one dimension.  Set this expression equal "
            "to zero and solve for f(x) with an appropriate potential V."
        ),
        "variables": "x",
        "symbols_used": ["hbar", "m", "E", "V"],
        "tags": ["schrodinger", "fundamental", "1d"],
    },
    "time_dependent_schrodinger_1d": {
        "name": "Time-Dependent Schrodinger Equation (1D)",
        "expression": (
            "I*hbar*f(x,t).diff(t) "
            "+ hbar**2/(2*m)*f(x,t).diff(x,2) - V*f(x,t)"
        ),
        "description": (
            "i*hbar * d psi/dt = H psi in 1D.  Expression equals zero "
            "when the equation is satisfied."
        ),
        "variables": "x,t",
        "symbols_used": ["hbar", "m", "V"],
        "tags": ["schrodinger", "fundamental", "time_dependent", "1d"],
    },
    "radial_schrodinger_equation": {
        "name": "Radial Schrodinger Equation",
        "expression": (
            "-hbar**2/(2*m) * (f(r).diff(r,2) + 2/r * f(r).diff(r)) "
            "+ (hbar**2*l*(l+1))/(2*m*r**2)*f(r) + V*f(r) - E*f(r)"
        ),
        "description": (
            "Radial part of the 3D Schrodinger equation in spherical "
            "coordinates for R(r).  Set equal to zero."
        ),
        "variables": "r",
        "symbols_used": ["hbar", "m", "l", "E", "V"],
        "tags": ["schrodinger", "radial", "3d", "spherical"],
    },

    # ── Potentials ───────────────────────────────────────────────
    "harmonic_oscillator_potential": {
        "name": "Quantum Harmonic Oscillator Potential",
        "expression": "Rational(1,2)*m*omega**2*x**2",
        "description": "V(x) = (1/2) m omega^2 x^2",
        "variables": "x",
        "symbols_used": ["m", "omega"],
        "tags": ["potential", "harmonic_oscillator"],
    },
    "coulomb_potential": {
        "name": "Coulomb Potential",
        "expression": "-Z*e**2/(4*pi*epsilon_0*r)",
        "description": (
            "V(r) = -Z e^2 / (4 pi epsilon_0 r) for hydrogen-like atoms."
        ),
        "variables": "r",
        "symbols_used": ["Z", "r"],
        "tags": ["potential", "coulomb", "hydrogen"],
    },
    "coulomb_potential_natural": {
        "name": "Coulomb Potential (Natural / Atomic Units)",
        "expression": "-Z/r",
        "description": "V(r) = -Z/r in atomic units (e = 4 pi epsilon_0 = 1).",
        "variables": "r",
        "symbols_used": ["Z", "r"],
        "tags": ["potential", "coulomb", "hydrogen", "atomic_units"],
    },
    "infinite_square_well_potential": {
        "name": "Infinite Square Well Potential",
        "expression": "Piecewise((0, (x >= 0) & (x <= a)), (oo, True))",
        "description": (
            "V(x) = 0 for 0 <= x <= a, infinity otherwise "
            "(particle in a box)."
        ),
        "variables": "x",
        "symbols_used": ["a"],
        "tags": ["potential", "square_well", "particle_in_box"],
    },
    "finite_square_well_potential": {
        "name": "Finite Square Well Potential",
        "expression": "Piecewise((-V0, (x >= -a) & (x <= a)), (0, True))",
        "description": "V(x) = -V0 for |x| <= a, 0 otherwise.",
        "variables": "x",
        "symbols_used": ["V0", "a"],
        "tags": ["potential", "square_well", "finite"],
    },
    "step_potential": {
        "name": "Step Potential",
        "expression": "Piecewise((0, x < 0), (V0, x >= 0))",
        "description": "V(x) = 0 for x < 0, V0 for x >= 0.",
        "variables": "x",
        "symbols_used": ["V0"],
        "tags": ["potential", "step", "scattering"],
    },
    "delta_function_potential": {
        "name": "Delta Function Potential",
        "expression": "-V0 * DiracDelta(x)",
        "description": (
            "V(x) = -V0 delta(x), an attractive delta-function potential."
        ),
        "variables": "x",
        "symbols_used": ["V0"],
        "tags": ["potential", "delta_function"],
    },

    # ── Operators ────────────────────────────────────────────────
    "momentum_operator_1d": {
        "name": "Momentum Operator (1D)",
        "expression": "-I*hbar*Derivative(f(x), x)",
        "description": "p_hat = -i hbar d/dx",
        "variables": "x",
        "symbols_used": ["hbar"],
        "tags": ["operator", "momentum"],
    },
    "kinetic_energy_operator_1d": {
        "name": "Kinetic Energy Operator (1D)",
        "expression": "-hbar**2/(2*m) * Derivative(f(x), x, 2)",
        "description": "T_hat = -hbar^2 / (2m) d^2/dx^2",
        "variables": "x",
        "symbols_used": ["hbar", "m"],
        "tags": ["operator", "kinetic_energy"],
    },

    # ── Eigenvalues / relations ──────────────────────────────────
    "angular_momentum_squared": {
        "name": "Angular Momentum Squared Eigenvalue",
        "expression": "hbar**2 * l*(l+1)",
        "description": "Eigenvalue of L^2: hbar^2 l(l+1).",
        "variables": "",
        "symbols_used": ["hbar", "l"],
        "tags": ["angular_momentum", "eigenvalue"],
    },
    "angular_momentum_z": {
        "name": "Angular Momentum z-component Eigenvalue",
        "expression": "hbar * m_l",
        "description": "Eigenvalue of L_z: hbar m_l.",
        "variables": "",
        "symbols_used": ["hbar"],
        "tags": ["angular_momentum", "eigenvalue"],
    },

    # ── Hydrogen-specific ────────────────────────────────────────
    "hydrogen_radial_equation": {
        "name": "Hydrogen Atom Radial Equation (reduced)",
        "expression": (
            "f(r).diff(r,2) + 2/r*f(r).diff(r) "
            "- l*(l+1)/r**2*f(r) "
            "+ 2*m/(hbar**2)*(E + Z/r)*f(r)"
        ),
        "description": (
            "Radial Schrodinger equation for hydrogen-like atoms.  "
            "Set equal to zero."
        ),
        "variables": "r",
        "symbols_used": ["l", "m", "hbar", "E", "Z"],
        "tags": ["hydrogen", "radial", "schrodinger"],
    },
}


# ── Internal helper (used by symbolic_math) ─────────────────────

def resolve_expression(key: str):
    """Return the raw SymPy expression string for a catalog key.

    This is an *internal* function — it is NOT exposed to the model.
    Only ``symbolic_math.substitute`` should call it.
    """
    entry = EQUATION_CATALOG.get(key)
    return entry["expression"] if entry else None


def _find_key(name: str):
    """Exact-or-fuzzy match a name to a catalog key."""
    if name in EQUATION_CATALOG:
        return name
    name_lower = name.lower().replace(" ", "_").replace("-", "_")
    matches = [
        key for key, entry in EQUATION_CATALOG.items()
        if name_lower in key
        or name_lower in entry["name"].lower().replace(" ", "_")
    ]
    return matches[0] if len(matches) == 1 else None


# ── Tool handler ─────────────────────────────────────────────────

def run_lookup_equation(
    operation: str,
    name: str = "",
    tag: str = "",
) -> dict:
    """Look up equations from the built-in catalog.

    The ``get`` operation returns metadata (name, description,
    variables, symbols, tags) but **not** the raw expression.
    To use the expression, pass its key to
    ``symbolic_math(operation="substitute", ...)``.

    Args:
        operation: "list" to see available equations, "get" to
                   retrieve metadata for one.
        name:      [get] Canonical key of the equation.
        tag:       [list] Optional tag to filter results.

    Returns:
        dict with keys: "result", "success", "error"
    """
    try:
        if operation == "list":
            entries = []
            for key, entry in EQUATION_CATALOG.items():
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
                entry = EQUATION_CATALOG[resolved_key]
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
                key for key, entry in EQUATION_CATALOG.items()
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
                available = list(EQUATION_CATALOG.keys())
                return {
                    "result": None,
                    "success": False,
                    "error": (
                        f"Equation '{name}' not found. "
                        f"Available keys: {available}"
                    ),
                }

        else:
            return {
                "result": None,
                "success": False,
                "error": f"Unknown operation: {operation}. Use 'list' or 'get'.",
            }

    except Exception as e:
        return {
            "result": None,
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
        }
