"""Tool JSON schemas for the Anthropic API."""

TOOLS = [
    {
        "name": "symbolic_math",
        "description": (
            "Perform symbolic mathematics using SymPy. Results are "
            "stored in a registry and you receive the key, the "
            "expression string, and a description. Chain operations "
            "by passing expression_key to the next call.\n\n"
            "IMPORTANT: All inputs must use expression_key (a "
            "registry or catalog key). You cannot pass raw "
            "expression strings — use lookup_equation and "
            "substitute to build expressions.\n\n"
            "ASSUMPTIONS: Call set_assumptions early in your "
            "workflow to declare symbol properties "
            "(e.g. assume_real=['x','m'], "
            "assume_positive=['E']). These are passed to "
            "SymPy and affect dsolve, simplify, and all "
            "operations.\n\n"
            "Operations and their key parameters:\n"
            "- list_stored: no params — list all stored expressions "
            "with their keys, descriptions, and expression strings.\n"
            "- set_assumptions: assume_real, assume_positive, "
            "assume_negative, assume_integer, assume_nonnegative, "
            "assume_nonpositive, assume_even, assume_odd — declare "
            "symbol properties for this session. Call once early.\n"
            "- clear_assumptions: no params — reset assumptions.\n"
            "- substitute: equation_key OR expression_key, "
            "substitutions — substitute symbol values into a catalog "
            "equation or a registry expression. Each substitution "
            "value is '@key' (registry or catalog) or a simple "
            "value string.\n"
            "- solve: expression_key, variable\n"
            "- solveset: expression_key, variable, domain — like "
            "solve but returns all solution families (e.g. n*pi/a "
            "for sin equations). Use for transcendental/periodic "
            "equations.\n"
            "- integrate: expression_key, variable, diff_variable, "
            "lower_limit, upper_limit\n"
            "- differentiate: expression_key, variable, "
            "diff_variable, order\n"
            "- eigenvalues: expression_key (Matrix)\n"
            "- solve_ode: expression_key (ODE), variable, hint, "
            "boundary_conditions (\"normalizable\" or "
            "[{point, value}])\n"
            "- simplify: expression_key\n"
            "- commutator: expression_key (operator A), "
            "operator_b_key, variable, diff_variable\n"
            "- time_evolution: expression_key (initial state), "
            "energy OR energy_key OR hamiltonian_key\n"
            "- fourier_transform: expression_key, variable, "
            "transform_variable, convention\n"
            "- limit: expression_key, variable, point, direction "
            "(optional: '+' or '-' for one-sided). "
            "Computes sp.limit(expr, var, point).\n"
            "- asymptotic: expression_key, variable, point. "
            "Decomposes expression into additive terms and "
            "reports the leading asymptotic behavior of each "
            "term as variable approaches point. Returns "
            "{terms: [{term, leading_behavior, behavior}]} "
            "where behavior is 'diverges', 'decays', 'finite', "
            "or 'unknown'. Use this to determine which terms "
            "in an ODE solution are physically acceptable.\n"
            "- set_equal: expression_key, value (default '0'). "
            "Creates Eq(expression, value). Use to form "
            "equations for solve (e.g. set integral = 1 for "
            "normalization).\n"
            "- arithmetic: expression_key, op, second_key "
            "(for binary ops). Perform arithmetic on stored "
            "expressions. Unary ops (expression_key only): "
            "conjugate, abs_squared, negate, sin, cos, exp, "
            "log, sqrt. Binary ops (expression_key + "
            "second_key): add, subtract, multiply, divide, "
            "power. Example: abs_squared on a wavefunction "
            "to get probability density.\n"


            "Returns: {\"stored_as\": \"expr_N\", "
            "\"expression\": \"<SymPy string>\", "
            "\"description\": \"...\", \"success\": bool, "
            "\"error\": str | null}  (list_stored returns "
            "{\"result\": [{\"key\", \"description\", "
            "\"expression\"}, ...]})"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "list_stored", "set_assumptions",
                        "clear_assumptions", "substitute",
                        "solve", "integrate", "differentiate",
                        "eigenvalues", "solve_ode", "simplify",
                        "commutator",
                        "time_evolution", "fourier_transform",
                        "arithmetic",
                        "limit", "set_equal", "asymptotic",
                        "solveset"
                    ],
                    "description": "The symbolic operation to perform."
                },
                "expression_key": {
                    "type": "string",
                    "description": (
                        "[all except list_stored] "
                        "Registry key (e.g. 'expr_1') or "
                        "catalog key to use as input expression. "
                        "For substitute, use this OR equation_key "
                        "(not both). "
                        "Use list_stored to see available registry "
                        "keys, or lookup_equation to find catalog keys."
                    )
                },
                "variable": {
                    "type": "string",
                    "description": (
                        "Variable name(s). "
                        "Single: 'x'. Multiple: 'r,theta,phi'. "
                        "Default: 'x'."
                    )
                },
                "diff_variable": {
                    "type": "string",
                    "description": (
                        "[integrate, differentiate, "
                        "commutator, fourier_transform] "
                        "Which variable to operate over when using "
                        "multiple variables. "
                        "Example: 'theta' when variable='r,theta,phi'."
                    )
                },
                "lower_limit": {
                    "type": "string",
                    "description": (
                        "[integrate] "
                        "Lower bound for integration. "
                        "Supports: numbers, symbols, '-oo', '-inf', "
                        "'-infinity'."
                    )
                },
                "upper_limit": {
                    "type": "string",
                    "description": (
                        "[integrate] "
                        "Upper bound for integration. "
                        "Supports: numbers, symbols, 'oo', 'inf', "
                        "'infinity'."
                    )
                },
                "order": {
                    "type": "integer",
                    "description": (
                        "[differentiate] Derivative order. Default: 1."
                    )
                },
                "hint": {
                    "type": "string",
                    "description": (
                        "[solve_ode] SymPy ODE solver hint."
                    )
                },
                "boundary_conditions": {
                    "description": (
                        "[solve_ode] Boundary conditions to apply "
                        "after solving. Pass \"normalizable\" for "
                        "open domains (drops divergent terms, "
                        "extracts quantization conditions). Or "
                        "pass a list of {\"point\": str, "
                        "\"value\": str} for finite domains "
                        "(e.g. [{\"point\": \"0\", \"value\": "
                        "\"0\"}, {\"point\": \"a\", \"value\": "
                        "\"0\"}])."
                    )
                },
                "operator_b_key": {
                    "type": "string",
                    "description": (
                        "[commutator] Registry key for operator B "
                        "in [A, B]. expression_key is operator A."
                    )
                },
                "energy": {
                    "type": "string",
                    "description": (
                        "[time_evolution] Energy eigenvalue as a "
                        "simple string. Example: 'hbar*omega/2'. "
                        "Alternative to energy_key."
                    )
                },
                "energy_key": {
                    "type": "string",
                    "description": (
                        "[time_evolution] Registry key for the "
                        "energy eigenvalue. Alternative to energy."
                    )
                },
                "hamiltonian_key": {
                    "type": "string",
                    "description": (
                        "[time_evolution] Registry key for the "
                        "Hamiltonian matrix. Use when energy is "
                        "not available."
                    )
                },
                "transform_variable": {
                    "type": "string",
                    "description": (
                        "[fourier_transform] Fourier conjugate variable. "
                        "Default: 'k'."
                    )
                },
                "convention": {
                    "type": "string",
                    "enum": ["physics", "math"],
                    "description": (
                        "[fourier_transform] Fourier convention. "
                        "'physics': 1/sqrt(2pi) * integral f(x) "
                        "exp(-ikx) dx. "
                        "'math': integral f(x) exp(-2pi*i*k*x) dx. "
                        "Default: 'physics'."
                    )
                },
                "equation_key": {
                    "type": "string",
                    "description": (
                        "[substitute] Catalog key from "
                        "lookup_equation, e.g. "
                        "'time_independent_schrodinger_1d'. "
                        "Use this for catalog equations, or use "
                        "expression_key for registry expressions."
                    )
                },
                "substitutions": {
                    "type": "object",
                    "description": (
                        "[substitute] Map of symbol name to "
                        "replacement value. Values can be:\n"
                        "- '@key': reference a registry key "
                        "(e.g. '@expr_1') or catalog key "
                        "(e.g. '@harmonic_oscillator_potential')\n"
                        "- A simple value: single number ('0', "
                        "'3.14') or single symbol name ('x', 'n')\n"
                        "Compound expressions are NOT allowed as "
                        "plain strings — use @key instead.\n"
                        "Example: {\"V\": "
                        "\"@harmonic_oscillator_potential\", "
                        "\"n\": \"0\"}"
                    )
                },
                "op": {
                    "type": "string",
                    "enum": [
                        "add", "subtract", "multiply", "divide",
                        "power", "conjugate", "abs_squared",
                        "negate", "sin", "cos", "tan", "exp", "log",
                        "sqrt"
                    ],
                    "description": (
                        "[arithmetic] The arithmetic operation. "
                        "Binary ops (need second_key): add, "
                        "subtract, multiply, divide, power. "
                        "Unary ops: conjugate, abs_squared, "
                        "negate, sin, cos, tan, exp, log, sqrt."
                    )
                },
                "second_key": {
                    "type": "string",
                    "description": (
                        "[arithmetic] Registry "
                        "key for the second operand in binary ops."
                    )
                },
                "point": {
                    "type": "string",
                    "description": (
                        "[limit, asymptotic] Target point "
                        "for limit evaluation. Examples: 'oo', "
                        "'-oo', 'a', '0', 'L'."
                    )
                },
                "direction": {
                    "type": "string",
                    "enum": ["+", "-"],
                    "description": (
                        "[limit] Direction for one-sided limits. "
                        "'+' for right-sided, '-' for left-sided. "
                        "Omit for two-sided limit."
                    )
                },
                "domain": {
                    "type": "string",
                    "enum": ["Reals", "Integers", "Complexes"],
                    "description": (
                        "[solveset] Domain to solve over. "
                        "'Reals' (default), 'Integers', or "
                        "'Complexes'."
                    )
                },
                "value": {
                    "type": "string",
                    "description": (
                        "[set_equal] Target value for the "
                        "equation. Creates Eq(expression, value). "
                        "Default: '0'. Example: '1' for "
                        "normalization (integral = 1)."
                    )
                },
                "assume_real": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "[set_assumptions] Symbol names to "
                        "assume real. Example: "
                        "['x','m','omega','hbar']."
                    )
                },
                "assume_positive": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "[set_assumptions] Symbol names to "
                        "assume positive. Implies real. "
                        "Example: ['E'] for bound states "
                        "with positive energy."
                    )
                },
                "assume_negative": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "[set_assumptions] Symbol names to "
                        "assume negative. Implies real."
                    )
                },
                "assume_integer": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "[set_assumptions] Symbol names to "
                        "assume integer."
                    )
                },
                "assume_nonnegative": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "[set_assumptions] Symbol names to "
                        "assume nonnegative (>= 0). "
                        "Implies real."
                    )
                },
                "assume_nonpositive": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "[set_assumptions] Symbol names to "
                        "assume nonpositive (<= 0). "
                        "Implies real."
                    )
                },
                "assume_even": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "[set_assumptions] Symbol names to "
                        "assume even integer."
                    )
                },
                "assume_odd": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "[set_assumptions] Symbol names to "
                        "assume odd integer."
                    )
                }
            },
            "required": ["operation"],
            "additionalProperties": False
        }
    },
    {
        "name": "numerical_compute",
        "description": (
            "Perform numerical computation using SciPy/NumPy. "
            "All operations require expression_key (registry key) "
            "— no raw strings accepted.\n\n"
            "Operations and their key parameters:\n"
            "- root_finding: expression_key, bracket OR x0, params\n"
            "- find_all_roots: expression_key, x_min, x_max, "
            "n_points, params\n"
            "- integrate_quad: expression_key, lower, upper, params\n"
            "- solve_ode: expression_key (ODE), x_span, y0, params, "
            "method. The tool auto-decomposes the ODE into a "
            "first-order system.\n"
            "- solve_bvp: expression_key (ODE), boundary_conditions, "
            "x_span, y_guess, params, expected_nodes\n"
            "- normalize_wavefunction: data_key (from solve_ode/"
            "solve_bvp), variable_index\n"
            "- evaluate_grid: expression_key (or expression_keys), "
            "x_min, x_max, n_points, variable, params\n\n"
            "IMPORTANT: Operations that produce arrays (solve_ode, "
            "solve_bvp, evaluate_grid, normalize_wavefunction) store "
            "results internally and return an opaque data key + "
            "summary (ranges, point count). Pass data keys to "
            "plot_results(data_keys=[...]) or "
            "normalize_wavefunction(data_key=...). Arrays are NEVER "
            "returned directly.\n\n"
            "Return formats (all include \"success\": bool, "
            "\"error\": str | null):\n"
            "- root_finding: {\"result\": {\"root\": float, "
            "\"verification\": float}}\n"
            "- find_all_roots: {\"result\": {\"roots\": [...], "
            "\"n_roots\": int}}\n"
            "- integrate_quad: {\"result\": {\"value\": float, "
            "\"absolute_error\": float}}\n"
            "- solve_ode/solve_bvp: {\"result\": "
            "{\"stored_as\": \"data_N\", \"summary\": "
            "{\"n_points\", \"x_range\", \"y0_range\", ...}}}\n"
            "- normalize_wavefunction: {\"result\": "
            "{\"stored_as\": \"data_N\", \"original_norm\", "
            "\"normalized_norm\", \"summary\": {...}}}\n"
            "- evaluate_grid: {\"result\": "
            "{\"stored_as\": \"data_N\", \"summary\": {...}}}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "root_finding", "find_all_roots",
                        "integrate_quad", "solve_ode", "solve_bvp",
                        "normalize_wavefunction",
                        "evaluate_grid"
                    ],
                    "description": "The numerical operation to perform."
                },
                "expression_key": {
                    "type": "string",
                    "description": (
                        "[root_finding, find_all_roots, integrate_quad, "
                        "solve_ode, solve_bvp] "
                        "REQUIRED. Registry key for a symbolic expression. "
                        "For root_finding/find_all_roots/integrate_quad: "
                        "the function to evaluate numerically. "
                        "For solve_ode/solve_bvp: the ODE equation "
                        "(Eq or expression=0). The tool auto-decomposes "
                        "it into a first-order system. "
                        "Piecewise expressions (from piecewise "
                        "potentials) are supported."
                    )
                },
                "x0": {
                    "type": "number",
                    "description": (
                        "[root_finding] Initial guess for fsolve. "
                        "Use bracket instead when possible."
                    )
                },
                "bracket": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "[root_finding] [a, b] bracket for brentq. "
                        "Preferred over x0."
                    )
                },
                "lower": {
                    "type": "number",
                    "description": (
                        "[integrate_quad] Lower integration bound. "
                        "Use -1e308 for -infinity."
                    )
                },
                "upper": {
                    "type": "number",
                    "description": (
                        "[integrate_quad] Upper integration bound. "
                        "Use 1e308 for infinity."
                    )
                },
                "x_span": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "[solve_ode, solve_bvp] "
                        "[x_start, x_end] integration interval."
                    )
                },
                "y0": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "[solve_ode] Initial conditions. "
                        "Example for 2nd order: [y(0), y'(0)]."
                    )
                },
                "boundary_conditions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "side": {
                                "type": "string",
                                "enum": ["left", "right"]
                            },
                            "variable_index": {
                                "type": "integer",
                                "description": (
                                    "0 for y(boundary), "
                                    "1 for y'(boundary), etc."
                                )
                            },
                            "value": {
                                "type": "number",
                                "description": (
                                    "Required value. Default: 0."
                                )
                            }
                        },
                        "required": ["side"]
                    },
                    "description": (
                        "[solve_bvp] Structured boundary conditions. "
                        "Example for psi(a)=0, psi(b)=0: "
                        "[{\"side\":\"left\",\"variable_index\":0,"
                        "\"value\":0},{\"side\":\"right\","
                        "\"variable_index\":0,\"value\":0}]"
                    )
                },
                "y_guess": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "description": (
                        "[solve_bvp] Initial guess. "
                        "Shape: (n_vars, n_points). "
                        "Use a sine or Gaussian shape for bound "
                        "states. Include derivative guess as "
                        "second row."
                    )
                },
                "params": {
                    "type": "object",
                    "description": (
                        "[all operations with expression_key] "
                        "Named physical constants. Used to "
                        "substitute symbol values in the expression. "
                        "Example: {\"hbar\": 1, \"m\": 1, "
                        "\"omega\": 2.0}"
                    )
                },
                "function_name": {
                    "type": "string",
                    "description": (
                        "[solve_ode, solve_bvp] Name of the "
                        "dependent function in the ODE. "
                        "Default: 'f'. Must match the function "
                        "name used in the SymPy expression."
                    )
                },
                "method": {
                    "type": "string",
                    "enum": [
                        "RK45", "RK23", "DOP853",
                        "Radau", "BDF", "LSODA"
                    ],
                    "description": (
                        "[solve_ode] Integration method. "
                        "Default: 'RK45'. "
                        "Use 'Radau' or 'BDF' for stiff problems."
                    )
                },
                "n_points": {
                    "type": "integer",
                    "description": (
                        "[find_all_roots, solve_ode, solve_bvp, "
                        "evaluate_grid] "
                        "Number of grid/output points. "
                        "Defaults: 500 (ODE), 100 (BVP), "
                        "1000 (roots), 500 (evaluate_grid)."
                    )
                },
                "expected_nodes": {
                    "type": "integer",
                    "description": (
                        "[solve_bvp] Expected zero crossings for "
                        "validation. Ground state: 0, "
                        "first excited: 1, etc."
                    )
                },
                "data_key": {
                    "type": "string",
                    "description": (
                        "[normalize_wavefunction] Data registry key "
                        "from solve_ode or solve_bvp. The tool "
                        "extracts the wavefunction and position "
                        "arrays automatically."
                    )
                },
                "variable_index": {
                    "type": "integer",
                    "description": (
                        "[normalize_wavefunction] Which y variable "
                        "to normalize (0 = wavefunction, "
                        "1 = derivative). Default: 0."
                    )
                },
                "expression_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "[evaluate_grid] List of registry keys "
                        "to evaluate on the grid. Alternative to "
                        "expression_key for multiple expressions."
                    )
                },
                "x_min": {
                    "type": "number",
                    "description": (
                        "[evaluate_grid, find_all_roots] "
                        "Lower bound. Default: -5."
                    )
                },
                "x_max": {
                    "type": "number",
                    "description": (
                        "[evaluate_grid, find_all_roots] "
                        "Upper bound. Default: 5."
                    )
                },
                "variable": {
                    "type": "string",
                    "description": (
                        "[evaluate_grid, solve_ode, solve_bvp] "
                        "Variable name. Default: 'x'."
                    )
                }
            },
            "required": ["operation"],
            "additionalProperties": False
        }
    },
    {
        "name": "plot_results",
        "description": (
            "Generate a publication-quality plot and save to "
            "outputs/figures/. Labels support LaTeX math with $...$.\n\n"
            "DATA INPUT (pick one):\n"
            "  1. data_keys: pass keys from numerical_compute "
            "(solve_ode, normalize_wavefunction, evaluate_grid). "
            "Use 'data_N:y1' to select a specific variable.\n"
            "  2. expression_keys: pass registry keys "
            "from symbolic_math + params for numerical values. "
            "The tool evaluates them on a grid automatically.\n"
            "  3. datasets: pass pre-computed {x, y} arrays.\n\n"
            "Example:\n"
            "  plot_results(\n"
            "    expression_keys=['expr_9','expr_10','expr_11'],\n"
            "    params={'hbar':1,'m':1,'omega':1},\n"
            "    labels=['n=0','n=1','n=2'],\n"
            "    title='Probability Densities',\n"
            "    x_label='Position x',\n"
            "    y_label='|psi_n(x)|^2',\n"
            "    x_min=-4, x_max=4)\n\n"
            "Returns: {\"filepath\": \"outputs/figures/plot_<timestamp>"
            ".png\", \"success\": bool, \"error\": str | null}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": (
                        "Plot title. Supports LaTeX $...$."
                    )
                },
                "x_label": {
                    "type": "string",
                    "description": (
                        "x-axis label. Supports LaTeX $...$."
                    )
                },
                "y_label": {
                    "type": "string",
                    "description": (
                        "y-axis label. Supports LaTeX $...$."
                    )
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Short, descriptive legend label for each "
                        "curve (e.g. 'n=0', 'n=1', 'Ground state',"
                        " 'V(x)'). Do NOT use expression keys "
                        "(expr_1) or full expressions as labels. "
                        "Supports LaTeX $...$."
                    )
                },
                "data_key": {
                    "type": "string",
                    "description": (
                        "Data registry key from numerical_compute "
                        "(e.g. 'data_1'). Use 'data_1:y1' to "
                        "select a specific variable."
                    )
                },
                "data_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of data registry keys to overlay. "
                        "Each can use 'data_N:yM' syntax."
                    )
                },
                "expression_key": {
                    "type": "string",
                    "description": (
                        "Registry key for a single expression "
                        "to evaluate and plot."
                    )
                },
                "expression_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of registry keys to evaluate and "
                        "overlay on the same plot."
                    )
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Parameter substitutions for expression "
                        "evaluation, e.g. {\"hbar\": 1, \"m\": 1,"
                        " \"omega\": 1}."
                    )
                },
                "x_min": {
                    "type": "number",
                    "description": (
                        "Grid start for expression evaluation. "
                        "Default: -5.0."
                    )
                },
                "x_max": {
                    "type": "number",
                    "description": (
                        "Grid end for expression evaluation. "
                        "Default: 5.0."
                    )
                },
                "n_points": {
                    "type": "integer",
                    "description": (
                        "Number of grid points for expression "
                        "evaluation. Default: 500."
                    )
                },
                "variable": {
                    "type": "string",
                    "description": (
                        "Variable name for expression evaluation."
                        " Default: 'x'."
                    )
                },
                "datasets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "array",
                                "items": {"type": "number"}
                            },
                            "y": {
                                "type": "array",
                                "items": {"type": "number"}
                            },
                            "label": {"type": "string"},
                            "style": {"type": "string"},
                            "color": {"type": "string"},
                            "fill": {"type": "boolean"}
                        },
                        "required": ["x", "y"]
                    },
                    "description": (
                        "Pre-computed data arrays. Each object "
                        "needs x and y arrays. Optional per-"
                        "dataset: label, style ('-','--',':'), "
                        "color, fill (bool)."
                    )
                },
                "hlines": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "y": {"type": "number"},
                            "label": {"type": "string"},
                            "color": {"type": "string"},
                            "style": {"type": "string"}
                        },
                        "required": ["y"]
                    },
                    "description": (
                        "Horizontal reference lines. Each needs "
                        "y value. Optional: label, color, style."
                    )
                },
                "fill": {
                    "type": "boolean",
                    "description": (
                        "Fill under all curves. Default: false. "
                        "Can also be set per-dataset."
                    )
                },
                "x_range": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "[x_min, x_max] axis limits."
                    )
                },
                "y_range": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "[y_min, y_max] axis limits."
                    )
                },
                "figsize": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "[width, height] in inches. "
                        "Default: [9, 5]."
                    )
                }
            },
            "required": ["title", "x_label", "y_label", "labels"],
            "additionalProperties": False
        }
    },
    {
        "name": "lookup_equation",
        "description": (
            "Look up well-known physics and quantum mechanics equations "
            "from the built-in catalog. Returns metadata (name, "
            "description, variables, symbols) but NOT the raw "
            "expression. To use an equation, pass its key to "
            "symbolic_math(operation=\"substitute\", "
            "equation_key=..., substitutions=...).\n\n"
            "Operations:\n"
            "- list: Show all available equations (optionally filtered "
            "by tag). Returns name, key, description, tags for each.\n"
            "- get: Retrieve metadata for a specific equation by its "
            "canonical key. Returns key, name, description, variables, "
            "symbols_used, tags, and usage instructions.\n"
            "- search: Semantic search over the catalog using natural "
            "language. Returns ranked results with relevance scores. "
            "Use this when you don't know the exact equation key.\n\n"
            "Returns: {\"result\": <list or dict>, \"success\": bool, "
            "\"error\": str | null}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["list", "get", "search"],
                    "description": (
                        "'list' to see available equations "
                        "(optionally filtered by tag), "
                        "'get' to retrieve a specific equation by key, "
                        "'search' for semantic search using natural language."
                    )
                },
                "name": {
                    "type": "string",
                    "description": (
                        "[get] Canonical key of the equation, e.g. "
                        "'harmonic_oscillator_potential', "
                        "'time_independent_schrodinger_1d'. "
                        "Use 'list' first if unsure of the exact key."
                    )
                },
                "tag": {
                    "type": "string",
                    "description": (
                        "[list/search] Optional tag to filter equations. "
                        "Examples: 'potential', 'schrodinger', "
                        "'hydrogen', 'operator', 'angular_momentum'."
                    )
                },
                "query": {
                    "type": "string",
                    "description": (
                        "[search] Natural-language search query. "
                        "Example: 'particle in a box', "
                        "'energy of hydrogen atom', "
                        "'magnetic field interaction'."
                    )
                },
                "n_results": {
                    "type": "integer",
                    "description": (
                        "[search] Maximum number of results to return. "
                        "Default: 5."
                    )
                }
            },
            "required": ["operation"],
            "additionalProperties": False
        }
    }
]
