"""Tests for the RAG pipeline: catalog loader, vector store, and search integration."""
import pytest

from tools.catalog_loader import get_catalog, reload_catalog
from tools.vector_store import search_equations
from tools.equations import run_lookup_equation, resolve_expression


# ── Catalog Loader ──────────────────────────────────────────────────

def test_catalog_loads_from_json():
    catalog = get_catalog()
    assert len(catalog) >= 17, f"Expected >= 17 equations, got {len(catalog)}"


ORIGINAL_KEYS = [
    "time_independent_schrodinger_1d",
    "time_dependent_schrodinger_1d",
    "radial_schrodinger_equation",
    "harmonic_oscillator_potential",
    "coulomb_potential",
    "coulomb_potential_natural",
    "infinite_square_well_potential",
    "finite_square_well_potential",
    "step_potential",
    "delta_function_potential",
    "momentum_operator_1d",
    "kinetic_energy_operator_1d",
    "hydrogen_radial_equation",
]


@pytest.mark.parametrize("key", ORIGINAL_KEYS)
def test_original_keys_present(key):
    catalog = get_catalog()
    assert key in catalog, f"Original key '{key}' missing from catalog"


def test_each_entry_has_required_fields():
    catalog = get_catalog()
    required = {"name", "expression", "description", "variables", "symbols_used", "tags"}
    for key, entry in catalog.items():
        missing = required - set(entry.keys())
        assert not missing, f"'{key}' missing fields: {missing}"


def test_each_entry_has_search_text():
    catalog = get_catalog()
    for key, entry in catalog.items():
        assert entry.get("search_text"), f"'{key}' missing search_text"


# ── Vector Store / Semantic Search ──────────────────────────────────

def test_search_particle_in_box():
    results = search_equations("particle in a box")
    assert len(results) > 0
    top_key = results[0]["key"]
    assert top_key == "infinite_square_well_potential", (
        f"Expected 'infinite_square_well_potential' as top hit, got '{top_key}'"
    )


def test_search_results_have_no_expression():
    results = search_equations("harmonic oscillator")
    for hit in results:
        assert "expression" not in hit, (
            f"Expression leaked in search result for '{hit['key']}'"
        )


def test_search_results_have_required_fields():
    results = search_equations("angular momentum")
    required = {"key", "name", "description", "tags", "score"}
    for hit in results:
        missing = required - set(hit.keys())
        assert not missing, f"Search hit missing fields: {missing}"


def test_search_tag_filtering():
    results = search_equations("energy", tag="spin")
    assert len(results) > 0
    for hit in results:
        assert "spin" in [t.lower() for t in hit["tags"]], (
            f"Hit '{hit['key']}' does not have 'spin' tag"
        )


def test_search_n_results():
    results = search_equations("quantum", n_results=3)
    assert len(results) <= 3


# ── Lookup Tool Integration ─────────────────────────────────────────

def test_list_operation():
    result = run_lookup_equation("list")
    assert result["success"]
    assert len(result["result"]) >= 17


def test_list_with_tag_filter():
    result = run_lookup_equation("list", tag="potential")
    assert result["success"]
    for entry in result["result"]:
        assert "potential" in [t.lower() for t in entry["tags"]]


def test_get_operation():
    result = run_lookup_equation("get", name="coulomb_potential")
    assert result["success"]
    assert result["result"]["key"] == "coulomb_potential"
    assert "expression" not in result["result"]


def test_search_operation():
    result = run_lookup_equation("search", query="particle in a box")
    assert result["success"]
    assert len(result["result"]) > 0
    assert result["result"][0]["key"] == "infinite_square_well_potential"


def test_search_operation_requires_query():
    result = run_lookup_equation("search")
    assert not result["success"]
    assert "query" in result["error"].lower()


# ── Resolve Expression (internal) ───────────────────────────────────

def test_resolve_expression_returns_string():
    expr = resolve_expression("harmonic_oscillator_potential")
    assert isinstance(expr, str)
    assert "omega" in expr


def test_resolve_expression_missing_key():
    assert resolve_expression("nonexistent_key") is None
