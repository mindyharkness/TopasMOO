"""
Pareto front visualization: 2D front, and pair projections for 3+ objectives.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ..metrics import calculate_knee_point
from .style import (
    ACCENT_COLOR,
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

# Light gray for context (other and infeasible designs) behind the front.
CONTEXT_GRAY = "0.78"


def objective_names_or_default(objective_names, n_objectives: int) -> list[str]:
    """``objective_names`` as a list, or "Objective 1", "Objective 2", ..."""
    if objective_names is None:
        return [f"Objective {i + 1}" for i in range(n_objectives)]
    if len(objective_names) != n_objectives:
        raise ValueError(
            f"objective_names must contain {n_objectives} labels; got {len(objective_names)}."
        )
    return list(objective_names)


def display_objectives(objectives: np.ndarray, maximize: Sequence[bool] | None) -> np.ndarray:
    """Flip the sign of maximized objectives for display.

    TopasMOO minimizes every objective, so a maximized quantity is returned
    negated by the objective function. Plots show it in its physical sense.
    Selection (knee point, ordering) must happen before this flip.
    """
    objectives = np.asarray(objectives, dtype=float)
    if maximize is None:
        return objectives
    return objectives * np.where(np.asarray(maximize, dtype=bool), -1.0, 1.0)


def _knee_point(ax, x, y) -> None:
    ax.scatter(x, y, s=marker_area(2.4), marker="*", c=ACCENT_COLOR, linewidths=0,
               label="Knee point", zorder=4)


def plot_pareto_front(
    pareto_objectives: np.ndarray,
    save_path: PathLike | None = None,
    *,
    objective_names: Sequence[str] | None = None,
    maximize: Sequence[bool] | None = None,
    evaluated_objectives: np.ndarray | None = None,
    eligible: np.ndarray | None = None,
    show_knee_point: bool = True,
    dpi: int | None = None,
):
    """Plot a Pareto front: 2D for two objectives, pair projections for more.

    :param pareto_objectives: Array of shape ``(n_solutions, n_objectives)``,
        minimized.
    :param save_path: If provided, save ``.pdf`` and ``.png`` to this path.
    :param objective_names: One axis label per objective (include units).
    :param maximize: One flag per objective; ``True`` shows that objective with
        its sign flipped (see :func:`display_objectives`).
    :param evaluated_objectives: All evaluated designs, drawn as gray context
        (two objectives only; see :func:`plot_pareto_front_2d`).
    :param eligible: Mask over ``evaluated_objectives`` of feasible, successful
        designs.
    :param show_knee_point: Mark the knee point.
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: The Axes (2D) or array of Axes (projections).
    """
    objectives = np.asarray(pareto_objectives, dtype=float)
    if objectives.ndim != 2 or objectives.shape[1] < 2:
        raise ValueError(
            "pareto_objectives must have shape (n_solutions, n_objectives) "
            f"with n_objectives >= 2; got shape {objectives.shape}."
        )
    common = dict(objective_names=objective_names, maximize=maximize,
                  show_knee_point=show_knee_point, dpi=dpi)
    if objectives.shape[1] == 2:
        return plot_pareto_front_2d(objectives, save_path, evaluated_objectives=evaluated_objectives,
                                    eligible=eligible, **common)
    return plot_pareto_front_projections(objectives, save_path, **common)


def plot_pareto_front_2d(
    pareto_objectives: np.ndarray,
    save_path: PathLike | None = None,
    *,
    ax: Axes | None = None,
    objective_names: Sequence[str] | None = None,
    maximize: Sequence[bool] | None = None,
    evaluated_objectives: np.ndarray | None = None,
    eligible: np.ndarray | None = None,
    true_front: np.ndarray | None = None,
    show_knee_point: bool = True,
    dpi: int | None = None,
) -> Axes:
    """Two-objective Pareto front with its knee point.

    Passing ``evaluated_objectives`` adds every other design as gray context:
    feasible designs as dots, infeasible ones as crosses. The axes then frame
    the front and the feasible designs, so far-off infeasible or failed
    evaluations are cropped rather than squeezing the front into a corner.

    :param pareto_objectives: Array of shape ``(n_solutions, 2)``, minimized.
    :param save_path: If provided, save ``.pdf`` and ``.png`` to this path.
    :param ax: Optional Axes to draw into (the figure is then not saved).
    :param objective_names: Two axis labels (include units).
    :param maximize: Per-objective display flags (see :func:`display_objectives`).
    :param evaluated_objectives: Optional ``(n_evaluations, 2)`` array of all
        evaluated designs, minimized.
    :param eligible: Mask over ``evaluated_objectives`` of feasible, successful
        designs (all ``True`` if omitted).
    :param true_front: Optional reference front, drawn dashed.
    :param show_knee_point: Mark the knee point.
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: The matplotlib Axes.
    """
    front = np.asarray(pareto_objectives, dtype=float).reshape(-1, 2)
    names = objective_names_or_default(objective_names, 2)
    shown = display_objectives(front, maximize)

    apply_style()
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=scale_figsize(SINGLE_COL_WIDTH, 2.8))
    else:
        fig = ax.figure

    context = evaluated_objectives is not None
    if context:
        evaluated = display_objectives(np.asarray(evaluated_objectives).reshape(-1, 2), maximize)
        ok = np.ones(len(evaluated), bool) if eligible is None else np.asarray(eligible, bool)
        ax.scatter(*evaluated[ok].T, s=marker_area(0.6), c=CONTEXT_GRAY, linewidths=0,
                   label="Other designs", zorder=1)
        if (~ok).any():
            ax.scatter(*evaluated[~ok].T, s=marker_area(0.7), marker="x", c=CONTEXT_GRAY,
                       linewidths=line_width(0.4), label="Infeasible", zorder=1)
        framed = np.vstack([shown, evaluated[ok]])
        if len(framed):
            lo, hi = framed.min(axis=0), framed.max(axis=0)
            pad = 0.05 * np.where(hi > lo, hi - lo, 1.0)
            ax.set(xlim=(lo[0] - pad[0], hi[0] + pad[0]), ylim=(lo[1] - pad[1], hi[1] + pad[1]))

    if true_front is not None:
        ax.plot(*np.asarray(true_front, dtype=float).T, ls="--", c="0.35", lw=line_width(0.7),
                label="True front", zorder=2)
    ax.scatter(*shown.T, s=marker_area(1.0), c="C0", linewidths=0, label="Pareto front", zorder=3)
    if show_knee_point and len(front):
        _knee_point(ax, *shown[calculate_knee_point(front)])

    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    format_publication_axes(ax)
    ax.legend(loc="best", markerscale=1.4 if context else 1.0, frameon=context,
              framealpha=0.95, edgecolor="none")

    finalize_figure(fig, save_path, own_fig=own_fig, dpi=dpi)
    return ax


def plot_pareto_front_projections(
    pareto_objectives: np.ndarray,
    save_path: PathLike | None = None,
    *,
    objective_names: Sequence[str] | None = None,
    maximize: Sequence[bool] | None = None,
    show_knee_point: bool = True,
    dpi: int | None = None,
) -> np.ndarray:
    """One 2D panel per objective pair, colored by another objective.

    Each panel is colored by the first objective not on its axes (for three
    objectives, the remaining one), with its own colorbar.

    :param pareto_objectives: Array of shape ``(n_solutions, n_objectives)``,
        ``n_objectives >= 3``, minimized.
    :param save_path: If provided, save ``.pdf`` and ``.png``.
    :param objective_names: One label per objective (include units).
    :param maximize: Per-objective display flags (see :func:`display_objectives`).
    :param show_knee_point: Mark the knee point in every panel.
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: Array of Axes, one per pair.
    """
    front = np.asarray(pareto_objectives, dtype=float)
    n_obj = front.shape[1]
    if n_obj < 3:
        raise ValueError(f"Projections need at least 3 objectives; got {n_obj}.")
    names = objective_names_or_default(objective_names, n_obj)
    shown = display_objectives(front, maximize)
    knee = calculate_knee_point(front) if show_knee_point and len(front) else None

    apply_style()
    pairs = [(i, j) for i in range(n_obj) for j in range(i + 1, n_obj)]
    n_cols = min(3, len(pairs))
    n_rows = int(np.ceil(len(pairs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                             figsize=scale_figsize(SINGLE_COL_WIDTH * n_cols, 3.0 * n_rows))
    axes = axes.flatten()
    for ax, (i, j) in zip(axes, pairs, strict=False):
        k = next(c for c in range(n_obj) if c not in (i, j))
        points = ax.scatter(shown[:, i], shown[:, j], c=shown[:, k], cmap="viridis",
                            s=marker_area(1.0), linewidths=0, zorder=3)
        if knee is not None:
            _knee_point(ax, shown[knee, i], shown[knee, j])
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        format_publication_axes(ax)
        fig.colorbar(points, ax=ax, location="top", fraction=0.05, pad=0.02).set_label(names[k])
    for ax in axes[len(pairs):]:
        ax.set_visible(False)

    finalize_figure(fig, save_path, dpi=dpi)
    return axes[: len(pairs)]
