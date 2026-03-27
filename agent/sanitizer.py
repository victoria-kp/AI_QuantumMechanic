"""Post-processing sanitizer for AI model output.

Strips mathematical notation that the model writes from memory,
including LaTeX, Unicode math symbols, and inline equations.
Applied to every AI message before it enters the conversation history.
"""
import re


# Unicode math symbols the model likes to use from memory
_UNICODE_MATH = {
    "ℏ": "hbar",
    "ψ": "psi",
    "Ψ": "Psi",
    "φ": "phi",
    "Φ": "Phi",
    "ω": "omega",
    "Ω": "Omega",
    "∂": "d",
    "∇": "nabla",
    "∫": "[integral]",
    "∑": "[sum]",
    "∏": "[product]",
    "√": "sqrt",
    "∞": "infinity",
    "π": "pi",
    "θ": "theta",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ε": "epsilon",
    "δ": "delta",
    "Δ": "Delta",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "σ": "sigma",
    "τ": "tau",
    "κ": "kappa",
    "χ": "chi",
    "ρ": "rho",
}

# Superscript/subscript digits
_SUPER_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ", "0123456789+-=()n")
_SUB_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₙ", "0123456789+-=()n")

# LaTeX patterns
_LATEX_DISPLAY = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_LATEX_INLINE = re.compile(r"\$[^$]+?\$")
_LATEX_FRAC = re.compile(r"\\frac\{[^}]*\}\{[^}]*\}")
_LATEX_COMMANDS = re.compile(r"\\[a-zA-Z]+(\{[^}]*\})*")

# Derivative patterns (Unicode and ASCII forms)
_DERIVATIVE_UNICODE = re.compile(r"d[²³]?[a-zA-Zψφ]/d[a-z][²³]?")
_DERIVATIVE_ASCII = re.compile(r"\bd[23]?[a-zA-Z]+/d[a-z][23]?\b")

# Equation-like lines (Unicode form — run before Unicode replacement)
_EQUATION_LINE_UNICODE = re.compile(
    r"^.*(?:"
    r"d².*?/dx²"        # second derivatives
    r"|d/dx"             # first derivatives
    r"|\\frac"           # LaTeX fractions
    r"|\\partial"        # partial derivatives
    r"|\bH\s*=\s*-"     # Hamiltonian definitions like H = -...
    r"|\bE_n\s*="        # eigenvalue formulas like E_n = ...
    r"|\bE\s*=\s*[-(]"  # energy formulas
    r").*$",
    re.MULTILINE,
)

# Equation-like lines (ASCII form — run after Unicode replacement)
# Catches lines containing physics-variable equations the model recalled
_EQUATION_LINE_ASCII = re.compile(
    r"^.*(?:"
    r"\bhbar\d*/\d*m\b"         # hbar/2m or hbar2/2m type fractions
    r"|\bpsi\b.*=.*\bpsi\b"    # psi ... = ... psi (eigenvalue eqs)
    r"|\bH\s+psi\s*="          # H psi = ...
    r"|\bE\s+psi\b"            # E psi (eigenvalue terms)
    r"|=\s*E\s+psi"            # ... = E psi
    r"|\bE_n\s*="              # E_n = ...
    r"|\bE\s*=\s*[-(n]"        # E = -(... or E = n... or E = (...
    r").*$",
    re.MULTILINE,
)

# Lines that look like standalone equations (contain = with physics terms
# on both sides, wrapped in ** bold markers)
_BOLD_EQUATION = re.compile(
    r"^\*\*[^*]*(?:"
    r"psi|hbar|omega|nabla|integral"
    r"|d[23]/d[a-z]"
    r"|derivative"
    r")[^*]*=.*\*\*\s*$",
    re.MULTILINE,
)


def sanitize_model_text(text):
    """Remove mathematical notation from model output text.

    Replaces:
    - LaTeX display/inline math ($...$, $$...$$)
    - Unicode math symbols (ℏ, ψ, ω, etc.) → word equivalents
    - Superscript/subscript Unicode digits → regular digits
    - Derivative notation (d²ψ/dx², d2psi/dx2)
    - Lines that look like recalled equations (both Unicode and ASCII)
    - Bold-wrapped equation lines

    Args:
        text: Raw model output string.

    Returns:
        Sanitized string with math notation removed.
    """
    if not text:
        return text

    # Strip LaTeX display math blocks
    text = _LATEX_DISPLAY.sub("[expression removed — use tool keys]", text)

    # Strip LaTeX inline math
    text = _LATEX_INLINE.sub("[expression removed]", text)

    # Strip remaining LaTeX commands
    text = _LATEX_FRAC.sub("[fraction removed]", text)
    text = _LATEX_COMMANDS.sub("", text)

    # Remove derivative notation and equation-like lines BEFORE Unicode
    # replacement (so patterns still match Unicode forms like d²ψ/dx²)
    text = _DERIVATIVE_UNICODE.sub("[derivative]", text)
    text = _EQUATION_LINE_UNICODE.sub("[recalled equation removed — use tools]", text)

    # Replace Unicode math symbols with words
    for sym, word in _UNICODE_MATH.items():
        text = text.replace(sym, word)

    # Replace superscript/subscript digits
    text = text.translate(_SUPER_DIGITS)
    text = text.translate(_SUB_DIGITS)

    # Second pass: catch equation-like lines in ASCII form
    text = _DERIVATIVE_ASCII.sub("[derivative]", text)
    text = _EQUATION_LINE_ASCII.sub("[recalled equation removed — use tools]", text)
    text = _BOLD_EQUATION.sub("[recalled equation removed — use tools]", text)

    return text
