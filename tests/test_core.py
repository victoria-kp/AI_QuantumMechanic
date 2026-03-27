"""Unit tests for expression registry, symbolic math, and equation catalog."""
import pytest
import sympy as sp

from AI_QuantumMechanic.tools import expression_registry as reg
from AI_QuantumMechanic.tools.symbolic_math import run_symbolic_math
from AI_QuantumMechanic.tools.equations import run_lookup_equation


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset the expression registry before each test."""
    reg.clear()


# ── Expression Registry ──────────────────────────────────────────────

def test_store_and_get():
    expr = sp.Symbol("x") ** 2
    key = reg.store(expr, "x squared")
    assert reg.get(key) == expr


def test_counter_increments():
    k1 = reg.store(sp.Integer(1), "one")
    k2 = reg.store(sp.Integer(2), "two")
    assert k1 == "expr_1"
    assert k2 == "expr_2"


def test_clear_resets():
    key = reg.store(sp.Integer(42), "forty-two")
    reg.clear()
    assert reg.get(key) is None
    new_key = reg.store(sp.Integer(0), "zero")
    assert new_key == "expr_1"


# ── Symbolic Math ────────────────────────────────────────────────────

def test_differentiate():
    # Store x**2, then differentiate
    key = reg.store(sp.Symbol("x") ** 2, "x squared")
    result = run_symbolic_math(operation="differentiate", expression_key=key)
    assert result["success"]
    assert "2*x" in result["expression"]


def test_integrate():
    key = reg.store(2 * sp.Symbol("x"), "2x")
    result = run_symbolic_math(operation="integrate", expression_key=key)
    assert result["success"]
    assert "x**2" in result["expression"]


def test_solve():
    # Solve x**2 - 4 = 0 -> x = -2, 2
    key = reg.store(sp.Symbol("x") ** 2 - 4, "x^2 - 4")
    result = run_symbolic_math(operation="solve", expression_key=key)
    assert result["success"]
    assert "solutions" in result  # multiple solutions


def test_substitute_from_catalog():
    result = run_symbolic_math(
        operation="substitute",
        equation_key="harmonic_oscillator_potential",
        substitutions={"x": "0"},
    )
    assert result["success"]
    assert result["stored_as"].startswith("expr_")


def test_substitute_rejects_compound_value():
    result = run_symbolic_math(
        operation="substitute",
        equation_key="harmonic_oscillator_potential",
        substitutions={"x": "hbar*omega"},
    )
    assert not result["success"]
    assert "compound" in result["error"].lower()


# ── Equation Catalog ─────────────────────────────────────────────────

def test_list_equations():
    result = run_lookup_equation(operation="list")
    assert result["success"]
    assert len(result["result"]) > 0
    assert all("key" in e for e in result["result"])


def test_get_equation():
    result = run_lookup_equation(
        operation="get", name="time_independent_schrodinger_1d"
    )
    assert result["success"]
    assert result["result"]["key"] == "time_independent_schrodinger_1d"
