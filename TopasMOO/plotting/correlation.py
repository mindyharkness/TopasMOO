"""
Parameter-objective correlation scatter plots.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

from .style import (
    SINGLE_COL_WIDTH,
    apply_style,
    finalize_figure,
    format_publication_axes,
    line_width,
    marker_area,
    scale_figsize,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]


def plot_parameter_objective_correlation(
    decision_vars: np.ndarray,
    objectives: np.ndarray,
    save_path: PathLike | None = None,
    *,
    parameter_names: Sequence[str] | None = None,
    objective_names: Sequence[str] | None = None,
    title: str = "Parameter-Objective Correlations",
    dpi: int | None = None,
) -> np.ndarray:
    """Grid of scatter plots: each parameter vs. each objective.

    :param decision_vars: Array of shape ``(n_solutions, n_params)``.
    :param objectives: Array of shape ``(n_solutions, n_objectives)``.
    :param save_path: If provided, save ``.pdf`` and ``.png``.
    :param parameter_names: Labels for parameters.
    :param objective_names: Labels for objectives.
    :param title: Figure super-title.
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: 2-D array of Axes.
    """
    apply_style()
    n_params = decision_vars.shape[1]
    n_obj = objectives.shape[1]

    fig, axes = plt.subplots(
        n_obj,
        n_params,
        figsize=scale_figsize(0.85 * SINGLE_COL_WIDTH * n_params, 2.5 * n_obj),
        squeeze=False,
    )

    for i in range(n_obj):
        for j in range(n_params):
            ax = axes[i, j]
            # A single color: coloring by the objective value would only
            # restate each panel's y-coordinate while squeezing the last column.
            ax.scatter(
                decision_vars[:, j],
                objectives[:, i],
                s=marker_area(0.85),
                alpha=0.6,
                edgecolors="black",
                linewidth=line_width(0.2),
            )
            if i == n_obj - 1:
                lbl = parameter_names[j] if parameter_names else f"Param {j + 1}"
                ax.set_xlabel(lbl)
            if j == 0:
                lbl = objective_names[i] if objective_names else f"Obj {i + 1}"
                ax.set_ylabel(lbl)
            format_publication_axes(ax)

    fig.suptitle(title)

    finalize_figure(fig, save_path, dpi=dpi)

    return axes
