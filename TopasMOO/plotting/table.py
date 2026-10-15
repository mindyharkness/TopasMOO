"""Solution table: the knee point and the best design for each objective."""

from __future__ import annotations

import os
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..metrics import calculate_knee_point
from .pareto import display_objectives, objective_names_or_default
from .style import DOUBLE_COL_WIDTH, apply_style, finalize_figure, scale_figsize, text_size

PathLike = Union[str, os.PathLike]


def select_solutions(pareto_objectives: np.ndarray, objective_names: Sequence[str]) -> dict[str, int]:
    """Row label -> index: the knee point, then the best design per objective.

    A design that is best for several objectives appears once ("Best f1, f3").
    """
    labels = {calculate_knee_point(pareto_objectives): "Knee point"}
    for j, name in enumerate(objective_names):
        i = int(np.argmin(pareto_objectives[:, j]))
        if i not in labels:
            labels[i] = f"Best {name}"
        elif labels[i].startswith("Best"):
            labels[i] += f", {name}"
    return {label: i for i, label in labels.items()}


def plot_solution_table(
    pareto_objectives: np.ndarray,
    pareto_decision_vars: np.ndarray,
    save_path: PathLike | None = None,
    *,
    objective_names: Sequence[str] | None = None,
    parameter_names: Sequence[str] | None = None,
    maximize: Sequence[bool] | None = None,
    dpi: int | None = None,
) -> Figure:
    """Table of the knee point and the best design per objective, real values only.

    :param pareto_objectives: ``(n_solutions, n_objectives)``, minimized.
    :param pareto_decision_vars: ``(n_solutions, n_params)``, row-aligned.
    :param save_path: If provided, save ``.pdf`` and ``.png``.
    :param objective_names: One label per objective.
    :param parameter_names: One label per parameter.
    :param maximize: Per-objective display flags.
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: The Figure.
    """
    F = np.asarray(pareto_objectives, dtype=float)
    X = np.asarray(pareto_decision_vars, dtype=float).reshape(len(F), -1)
    names = objective_names_or_default(objective_names, F.shape[1])
    params = list(parameter_names) if parameter_names is not None else [f"Parameter {k + 1}" for k in range(X.shape[1])]
    picks = select_solutions(F, names)
    shown = display_objectives(F, maximize)
    header = [""] + names + params
    rows = [[label] + [f"{v:.3g}" for v in np.r_[shown[i], X[i]]] for label, i in picks.items()]

    apply_style()
    n_lines = len(rows) + 1
    fig, ax = plt.subplots(figsize=scale_figsize(DOUBLE_COL_WIDTH, 0.3 * n_lines + 0.25))
    ax.set_axis_off()
    widths = np.r_[2.2, np.ones(len(header) - 1)]
    centers = (np.cumsum(widths) - widths / 2) / widths.sum()
    size = text_size("small")
    for r, line in enumerate([header] + rows):
        y = 1 - (r + 0.5) / n_lines
        for c, text in enumerate(line):
            ax.text(centers[c] if c else 0.0, y, text, transform=ax.transAxes, fontsize=size,
                    ha="center" if c else "left", va="center")
    for y, lw in ((1.0, 0.8), (1 - 1 / n_lines, 0.5), (0.0, 0.8)):
        ax.axhline(y, color="black", lw=lw)
    split = np.cumsum(widths)[len(names)] / widths.sum()
    ax.axvline(split, color="0.7", lw=0.5)
    finalize_figure(fig, save_path, dpi=dpi)
    return fig
