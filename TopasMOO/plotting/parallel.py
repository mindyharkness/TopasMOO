"""Parallel coordinates of the Pareto set in real units."""

from __future__ import annotations

import os
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ..metrics import calculate_knee_point
from .pareto import CONTEXT_GRAY, display_objectives, objective_names_or_default
from .style import (
    ACCENT_COLOR,
    DOUBLE_COL_WIDTH,
    apply_style,
    finalize_figure,
    line_width,
    scale_figsize,
    text_size,
)

PathLike = Union[str, os.PathLike]


def plot_parallel_coordinates(
    pareto_objectives: np.ndarray,
    save_path: PathLike | None = None,
    *,
    decision_vars: np.ndarray | None = None,
    lower_bounds: Sequence[float] | None = None,
    upper_bounds: Sequence[float] | None = None,
    objective_names: Sequence[str] | None = None,
    parameter_names: Sequence[str] | None = None,
    maximize: Sequence[bool] | None = None,
    show_knee_point: bool = True,
    ax: Axes | None = None,
    dpi: int | None = None,
) -> Axes:
    """Parallel coordinates: one gray line per Pareto solution, knee point highlighted.

    With ``decision_vars``, parameter axes come first and objective axes last,
    separated by a small gap. Every axis is labeled at its ends in real units:
    the parameter bounds (or the observed range when bounds are not given) and
    the range of the front for objectives.

    :param pareto_objectives: ``(n_solutions, n_objectives)``, minimized.
    :param save_path: If provided, save ``.pdf`` and ``.png``.
    :param decision_vars: Optional ``(n_solutions, n_params)``, row-aligned.
    :param lower_bounds: Optional parameter lower bounds (axis bottoms).
    :param upper_bounds: Optional parameter upper bounds (axis tops).
    :param objective_names: One label per objective.
    :param parameter_names: One label per parameter.
    :param maximize: Per-objective display flags.
    :param show_knee_point: Highlight the knee point.
    :param ax: Optional Axes to draw into (the figure is then not saved).
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: The matplotlib Axes.
    """
    F = np.asarray(pareto_objectives, dtype=float)
    shown = display_objectives(F, maximize)
    names = objective_names_or_default(objective_names, F.shape[1])
    columns, lo, hi, labels = [shown], [shown.min(axis=0)], [shown.max(axis=0)], list(names)
    n_params = 0
    if decision_vars is not None:
        X = np.asarray(decision_vars, dtype=float).reshape(len(F), -1)
        n_params = X.shape[1]
        columns.insert(0, X)
        lo.insert(0, X.min(axis=0) if lower_bounds is None else np.asarray(lower_bounds, dtype=float))
        hi.insert(0, X.max(axis=0) if upper_bounds is None else np.asarray(upper_bounds, dtype=float))
        labels = (list(parameter_names) if parameter_names is not None
                  else [f"Parameter {k + 1}" for k in range(n_params)]) + labels
    values, lo, hi = np.hstack(columns), np.concatenate(lo), np.concatenate(hi)
    normalized = (values - lo) / np.where(hi > lo, hi - lo, 1.0)
    xs = np.r_[np.arange(n_params), np.arange(F.shape[1]) + n_params + (0.6 if n_params else 0)]

    apply_style()
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=scale_figsize(DOUBLE_COL_WIDTH, 2.6))
    else:
        fig = ax.figure
    for row in normalized:
        ax.plot(xs, row, color=CONTEXT_GRAY, lw=line_width(0.4), zorder=1)
    if show_knee_point and len(F):
        ax.plot(xs, normalized[calculate_knee_point(F)], color=ACCENT_COLOR, lw=line_width(1.2),
                label="Knee point", zorder=2)
    small = text_size("small")
    for x, bottom, top in zip(xs, lo, hi, strict=True):
        ax.vlines(x, 0, 1, color="0.25", lw=line_width(0.4), zorder=0)
        ax.text(x, 1.03, f"{top:.3g}", ha="center", va="bottom", fontsize=small)
        ax.text(x, -0.03, f"{bottom:.3g}", ha="center", va="top", fontsize=small)
    ax.set_xticks(xs, labels)
    ax.tick_params(axis="x", length=0, pad=4 + small)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines[:].set_visible(False)
    ax.set_xlim(xs[0] - 0.4, xs[-1] + 0.4)
    ax.set_ylim(-0.1, 1.1)
    if show_knee_point and len(F):
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1.06), borderaxespad=0)
    finalize_figure(fig, save_path, own_fig=own_fig, dpi=dpi)
    return ax
