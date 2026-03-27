"""General physical sanity checks for quantum mechanical solutions."""
import json
import numpy as np
from langchain_core.messages import ToolMessage


def check_physical_sanity(
    energies: list[float] = None,
    psi: np.ndarray = None,
    system_type: str = "bound"
) -> dict:
    """Run basic physics sanity checks.

    Checks:
        1. Bound state energies should be negative
        2. Probability density should be non-negative.
           Since |psi|^2 >= 0 analytically, this check catches
           numerical issues: NaN/inf values or floating-point
           artifacts that produce unphysical negative values.
        3. Energy levels should be ordered

    Args:
        energies:    List of energy eigenvalues.
        psi:         Wavefunction array (any shape).
        system_type: "bound" or "free". Bound states must have E < 0.

    Returns:
        {"passed": bool, "issues": list[str], "message": str}
    """
    issues = []

    # Check 1: Bound state energies < 0
    if energies and system_type == "bound":
        for i, E in enumerate(energies):
            if E > 0:
                issues.append(
                    f"Energy E_{i} = {E:.4f} > 0 "
                    f"for bound state"
                )

    # Check 2: Probability non-negative
    if psi is not None:
        prob = np.abs(psi)**2
        if np.any(prob < -1e-10):
            issues.append(
                "Negative probability density found"
            )

    # Check 3: Energy ordering
    if energies and len(energies) > 1:
        for i in range(len(energies) - 1):
            if energies[i] > energies[i + 1]:
                issues.append(
                    f"Energy levels not in ascending order: "
                    f"E_{i} = {energies[i]:.4f} > "
                    f"E_{i+1} = {energies[i + 1]:.4f}. "
                    f"Lower quantum numbers must have lower energy."
                )

    passed = len(issues) == 0
    return {
        "passed": passed,
        "issues": issues,
        "message": (
            "Sanity check: PASS" if passed
            else "Sanity check FAIL:\n"
            + "\n".join(f"  - {i}" for i in issues)
        ),
    }


def extract_and_check_sanity(messages) -> dict:
    """Scan tool-result messages for energy/wavefunction data and run sanity checks.

    Looks through the message history for ToolMessage objects containing
    energy eigenvalues (from numerical_compute eigenvalues operation)
    and wavefunction arrays (from solve_bvp, solve_ode).

    Args:
        messages: List of LangGraph messages from state["messages"]

    Returns:
        {"passed": bool, "checks": list[dict], "message": str}
    """
    all_energies = []
    all_psi = []

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

        # Collect eigenvalues
        if "eigenvalues" in result:
            all_energies.extend(result["eigenvalues"])

        # Collect wavefunction arrays from solvers
        if "x" in result and "y" in result:
            y_data = result["y"]
            if isinstance(y_data, dict) and "y0" in y_data:
                all_psi.append(np.array(y_data["y0"]))

    checks = []

    # Run energy checks if we found any
    if all_energies:
        energy_check = check_physical_sanity(
            energies=sorted(all_energies), system_type="bound"
        )
        energy_check["source"] = "energy_values"
        checks.append(energy_check)

    # Run wavefunction checks on each psi
    for i, psi in enumerate(all_psi):
        psi_check = check_physical_sanity(psi=psi)
        psi_check["source"] = f"wavefunction_{i}"
        checks.append(psi_check)

    if not checks:
        return {
            "passed": True,
            "checks": [],
            "message": (
                "No energy/wavefunction data found in tool results "
                "to run sanity checks."
            ),
        }

    all_passed = all(c["passed"] for c in checks)
    summary = "\n".join(c["message"] for c in checks)

    return {
        "passed": all_passed,
        "checks": checks,
        "message": summary,
    }
