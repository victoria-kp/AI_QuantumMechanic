"""Plotting tool using matplotlib for the AI-QuantumMechanic agent."""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving files
import numpy as np
import os
from datetime import datetime
from pathlib import Path

_FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"

from .numerical_compute import _resolve_expression_key, get_data as _data_get
from .expression_registry import get_label as _reg_label


def _resolve_data_keys(kwargs):
    """Resolve data registry keys to plot datasets.

    Supports 'data_key:y1' syntax to select a specific variable from
    multi-variable ODE solutions.  Defaults to y0 (the wavefunction).
    """
    data_keys = kwargs.pop("data_keys", [])
    single_key = kwargs.pop("data_key", "")
    if single_key and not data_keys:
        data_keys = [single_key]
    if not data_keys:
        return None

    labels = kwargs.get("labels", [])
    datasets = []
    for idx, key in enumerate(data_keys):
        # Parse "data_1:y1" format
        parts = key.split(":")
        dkey = parts[0]
        var = parts[1] if len(parts) > 1 else "y0"

        data = _data_get(dkey)
        if data is None:
            raise ValueError(f"Unknown data key: '{dkey}'")

        x = np.asarray(data["x"])
        y_raw = data["y"]
        if isinstance(y_raw, dict):
            y = np.asarray(y_raw.get(var, y_raw.get("y0")))
        else:
            y = np.asarray(y_raw)

        label = labels[idx] if idx < len(labels) else key
        datasets.append({"x": x, "y": y, "label": label})

    return datasets


def _resolve_expression_data(kwargs):
    """If expression_keys/expression_key provided, evaluate on grid and
    build datasets list.  Returns (datasets, labels).

    Labels come from kwargs["labels"] — the LLM chooses them.
    expression_keys only determines WHAT data to evaluate.
    Falls back to the registry's short label if labels are missing.
    """
    expression_keys = kwargs.pop("expression_keys", [])
    single_key = kwargs.pop("expression_key", "")
    if single_key and not expression_keys:
        expression_keys = [single_key]
    if not expression_keys:
        return None, None  # nothing to resolve

    x_min = kwargs.pop("x_min", -5.0)
    x_max = kwargs.pop("x_max", 5.0)
    n_points = kwargs.pop("n_points", 500)
    variable = kwargs.pop("variable", "x")
    params = kwargs.pop("params", {})
    labels = kwargs.get("labels", [])

    x_grid = np.linspace(x_min, x_max, n_points)

    datasets = []
    for idx, key in enumerate(expression_keys):
        func = _resolve_expression_key(
            {"expression_key": key, "params": params},
            variable=variable,
        )
        y = np.empty_like(x_grid)
        for i, xi in enumerate(x_grid):
            try:
                val = float(func(xi))
                y[i] = val if np.isfinite(val) else 0.0
            except Exception:
                y[i] = 0.0
        # Use the LLM-provided label; fall back to registry short label
        label = labels[idx] if idx < len(labels) else _reg_label(key)
        datasets.append({
            "x": x_grid.tolist(),
            "y": y.tolist(),
            "label": label,
        })

    return datasets, labels


def run_plot(
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    labels: list = None,
    **kwargs,
) -> dict:
    """Generate a publication-quality plot and save to outputs/figures/.

    Data can be provided in two ways:
      1. expression_keys (list[str]) + params (dict)  — auto-evaluates
         symbolic expressions from the registry on a numerical grid.
      2. datasets (list[dict])  — pre-computed {x, y} arrays.

    Required:
        title, x_label, y_label, labels (legend label per curve).

    Optional:
        expression_keys, expression_key, params, x_min, x_max, n_points,
        variable, datasets, hlines, fill, x_range, y_range, figsize.

    Returns:
        dict with "filepath" (str), "success" (bool), "error" (str|None).
    """
    if labels is None:
        labels = []

    try:
        # --- Resolve data to datasets (priority: data_keys > expression_keys > datasets) ---
        # Forward labels into kwargs so resolvers can see them
        kwargs["labels"] = labels
        data_datasets = _resolve_data_keys(kwargs)
        expr_datasets, _ = _resolve_expression_data(kwargs)
        datasets = data_datasets or expr_datasets or kwargs.pop("datasets", [])

        # --- Warn on empty data ---
        if not datasets:
            return {
                "filepath": None,
                "success": False,
                "error": (
                    "No data to plot. Provide expression_keys (with "
                    "params) or datasets (list of {x, y} dicts)."
                ),
            }

        # --- Pad labels if too short ---
        while len(labels) < len(datasets):
            ds = datasets[len(labels)]
            labels.append(ds.get("label", f"curve_{len(labels)}"))

        # --- Sync labels into datasets ---
        for idx, ds in enumerate(datasets):
            ds["label"] = labels[idx]

        # --- Matplotlib setup ---
        plt.rcParams.update({
            'text.usetex': False,
            'mathtext.fontset': 'dejavusans',
            'font.family': 'serif',
        })
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("outputs", exist_ok=True)

        colors = [
            '#2563eb', '#dc2626', '#16a34a', '#9333ea',
            '#ea580c', '#0891b2', '#4f46e5', '#be123c',
        ]

        figsize = tuple(kwargs.get("figsize", [9, 5]))
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        fill_all = kwargs.get("fill", False)

        # --- Plot each dataset ---
        for i, ds in enumerate(datasets):
            x = np.array(ds["x"])
            y = np.array(ds["y"])
            lbl = ds.get("label", "")
            style = ds.get("style", "-")
            color = ds.get("color", colors[i % len(colors)])
            fill = ds.get("fill", fill_all)

            ax.plot(x, y, style, color=color, linewidth=2, label=lbl)
            if fill:
                ax.fill_between(x, y, alpha=0.2, color=color)

        # --- Horizontal reference lines ---
        for hline in kwargs.get("hlines", []):
            ax.axhline(
                y=hline.get("y", 0),
                color=hline.get("color", "gray"),
                linestyle=hline.get("style", "--"),
                linewidth=1.5,
                alpha=0.7,
                label=hline.get("label", ""),
            )

        # --- Axes and labels ---
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.set_xlabel(x_label, fontsize=13)
        ax.set_ylabel(y_label, fontsize=13)
        ax.set_title(title, fontsize=15)

        if kwargs.get("x_range"):
            ax.set_xlim(kwargs["x_range"])
        if kwargs.get("y_range"):
            ax.set_ylim(kwargs["y_range"])

        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        filepath = str(_FIGURES_DIR / f"plot_{timestamp}.png")
        fig.savefig(
            filepath, dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none',
        )
        plt.close(fig)

        return {"filepath": filepath, "success": True, "error": None}

    except Exception as e:
        plt.close('all')
        return {
            "filepath": None,
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
        }
