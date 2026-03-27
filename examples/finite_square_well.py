"""Finite Square Well — piecewise symbolic solve.

Demonstrates the agent's symbolic pipeline for piecewise potentials:
  lookup_equation -> substitute -> solve_ode (per region, normalizable BCs)
  -> substitute (boundary evaluation) -> differentiate -> arithmetic
  -> set_equal -> solve (matching conditions) -> find_all_roots
  -> substitute (energy) -> integrate -> set_equal -> solve (normalization)
  -> plot_results

The agent derives energy eigenvalues, normalized wavefunctions, and
probability densities for bound states entirely through tools.

Run from the repo root:
    python -m AI_QuantumMechanic.examples.finite_square_well
"""

import sys
from pathlib import Path

# Allow running as a standalone script from AI_QuantumMechanic/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from AI_QuantumMechanic.agent.graph import build_graph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# ── Problem statement ────────────────────────────────────────────────
PROBLEM = """\
Solve the Schrodinger equation for the finite square well potential \
with depth V0 and half-width a: V(x) = -V0 for |x| < a, V(x) = 0 otherwise. \
Find the first bound-state energy eigenvalue numerically (use hbar=1, m=1, a=1, V0=5). \
Plot the first bound-state wavefunction psi(x) from x = -5 to x = 5.\
"""

# ── Logging helper ───────────────────────────────────────────────────
LOG_FILE = Path(__file__).resolve().parent.parent / "outputs" / "logs" / "finite_square_well_log.txt"


def _log(text: str, *, _file_handle=[None]):
    """Print to stdout and append to log file."""
    if _file_handle[0] is None:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _file_handle[0] = open(LOG_FILE, "w")
        _file_handle[0].write("=== Finite Square Well — Agent Log ===\n\n")
    print(text)
    _file_handle[0].write(text + "\n")
    _file_handle[0].flush()


def _log_messages(msgs):
    for msg in msgs:
        if isinstance(msg, AIMessage):
            _log("\n  [AI]")
            if msg.content:
                _log(f"  {msg.content}")
            for tc in getattr(msg, "tool_calls", []):
                _log(f"\n  >> Tool call: {tc['name']}")
                _log(f"     Args: {tc['args']}")
        elif isinstance(msg, ToolMessage):
            _log(f"\n  [Tool result] (id: {msg.tool_call_id})")
            _log(f"  {msg.content}")
        elif isinstance(msg, HumanMessage):
            _log(f"\n  [Checker feedback]")
            _log(f"  {msg.content}")


# ── Run ──────────────────────────────────────────────────────────────
def main():
    app = build_graph()

    initial_state = {
        "messages": [HumanMessage(content=PROBLEM)],
        "stop_reason": "",
        "check_passed": False,
        "retry_count": 0,
        "figures": [],
        "tool_history": [],
    }

    # Stream and collect the final state
    final = dict(initial_state)
    for i, step in enumerate(app.stream(initial_state, config={"recursion_limit": 500})):
        for node_name, state_update in step.items():
            _log(f"\n{'=' * 60}")
            _log(f"Step {i}: {node_name}")
            _log("=" * 60)
            if "messages" in state_update:
                _log_messages(state_update["messages"])
            _log("")
            final.update(state_update)

    # ── Summary ──────────────────────────────────────────────────────
    _log("\n" + "=" * 60)
    _log("DONE")
    _log("=" * 60)
    _log(f"  Check passed : {final.get('check_passed')}")
    _log(f"  Stop reason  : {final.get('stop_reason')}")
    _log(f"  Tools used   : {len(final.get('tool_history', []))} calls")
    _log(f"  Figures      : {final.get('figures', [])}")
    _log(f"  Full log     : {LOG_FILE}")


if __name__ == "__main__":
    main()
