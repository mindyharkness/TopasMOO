"""Decision variables of the Pareto set, ordered along the front."""

from __future__ import annotations

import os
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from .pareto import objective_names_or_default
from .style import DOUBLE_COL_WIDTH, apply_style, finalize_figure, scale_figsize

PathLike = Union[str, os.PathLike]


def plot_decision_along_front(
    pareto_objectives: np.ndarray,
    pareto_decision_vars: np.ndarray,
    save_path: PathLike | None = None,
    *,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    parameter_names: Sequence[str] | None = None,
    objective_names: Sequence[str] | None = None,
    dpi: int | None = None,
) -> Axes:
    """Where each parameter sits within its bounds, for every Pareto solution.

    Columns are the Pareto solutions ordered by the first objective, rows are
    parameters, and color is the position between the lower (0) and upper (1)
    bound. A row that stays one color is fixed along the front; a gradient
    shows a parameter that drives the trade-off.

    :param pareto_objectives: ``(n_solutions, n_objectives)``, minimized.
    :param pareto_decision_vars: ``(n_solutions, n_params)``, row-aligned.
    :param save_path: If provided, save ``.pdf`` and ``.png``.
    :param lower_bounds: Lower bound of each parameter.
    :param upper_bounds: Upper bound of each parameter.
    :param parameter_names: One label per parameter.
    :param objective_names: One label per objective (the first names the order).
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: The matplotlib Axes.
    """
    F = np.asarray(pareto_objectives, dtype=float)
    X = np.asarray(pareto_decision_vars, dtype=float).reshape(len(F), -1)
    lower, upper = np.asarray(lower_bounds, dtype=float), np.asarray(upper_bounds, dtype=float)
    span = np.where(upper > lower, upper - lower, 1.0)
    position = (X[np.argsort(F[:, 0], kind="stable")] - lower) / span
    params = list(parameter_names) if parameter_names is not None else [f"Parameter {k + 1}" for k in range(X.shape[1])]
    first = objective_names_or_default(objective_names, F.shape[1])[0]

    apply_style()
    fig, ax = plt.subplots(figsize=scale_figsize(DOUBLE_COL_WIDTH, 1.2 + 0.32 * X.shape[1]))
    image = ax.imshow(position.T, aspect="auto", cmap="cividis", vmin=0, vmax=1,
                      interpolation="nearest")
    ax.grid(False)
    ax.set_yticks(range(len(params)), params)
    ax.xaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
    ax.set_xlabel(f"Pareto solutions, ordered by {first}")
    fig.colorbar(image, ax=ax, pad=0.02).set_label("Position within bounds")
    finalize_figure(fig, save_path, dpi=dpi)
    return ax
