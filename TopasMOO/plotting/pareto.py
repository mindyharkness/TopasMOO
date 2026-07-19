"""
Pareto front visualization (2D, 3D, and pairwise projections).
"""

from __future__ import annotations

import logging
import os
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ..metrics import (
    calculate_crowding_distance,
    calculate_dominance_rank,
    calculate_knee_point,
)
from .style import (
    ACCENT_COLOR,
    DOUBLE_COL_WIDTH,
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


def plot_pareto_front(
    pareto_objectives: np.ndarray,
    save_path: PathLike | None = None,
    *,
    show_knee_point: bool = False,
    dpi: int | None = None,
):
    """Plot a Pareto front, choosing 2D / 3D / projections from objective count.

    Shared dispatcher used by intermediate monitoring and the end-of-run
    comprehensive suite so dimension branching stays in one place.

    :param pareto_objectives: Array of shape ``(n_solutions, n_objectives)``.
    :param save_path: If provided, save ``.pdf`` and ``.png`` to this path.
    :param show_knee_point: Mark the knee point (2D and 3D only).
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: The matplotlib Axes (2D/3D) or Axes array (projections).
    """
    objectives = np.asarray(pareto_objectives, dtype=float)
    if objectives.ndim != 2 or objectives.shape[1] < 2:
        raise ValueError(
            "pareto_objectives must have shape (n_solutions, n_objectives) "
            f"with n_objectives >= 2; got shape {objectives.shape}."
        )
    n_obj = objectives.shape[1]
    if n_obj == 2:
        return plot_pareto_front_2d(
            objectives,
            save_path,
            show_knee_point=show_knee_point,
            dpi=dpi,
        )
    if n_obj == 3:
        return plot_pareto_front_3d(
            objectives,
            save_path,
            show_knee_point=show_knee_point,
            dpi=dpi,
        )
    return plot_pareto_front_projections(
        objectives,
        save_path,
        dpi=dpi,
    )


def _knee_point_style() -> dict:
    """Knee-point marker styling (orange star), shared by the 2D and 3D plots.

    Built per call so the marker scales with the active style's base sizes.
    """
    return dict(
        s=marker_area(2.9),
        marker="*",
        c=ACCENT_COLOR,
        edgecolors="#333333",
        linewidth=line_width(0.4),
        label="Knee point",
    )


def _resolve_metric(color_by_metric, metric_values, metric_label, objectives):
    """Resolve a ``color_by_metric`` request to ``(values, label)``.

    When ``metric_values`` is already provided it is returned unchanged. The
    named metrics ``"crowding"`` and ``"rank"`` are computed from ``objectives``
    with a sensible default label. Shared by the 2D and 3D Pareto plots.
    """
    if not color_by_metric or metric_values is not None:
        return metric_values, metric_label
    if color_by_metric == "crowding":
        return calculate_crowding_distance(objectives), metric_label or "Crowding Distance"
    if color_by_metric == "rank":
        return calculate_dominance_rank(objectives), metric_label or "Dominance Rank"
    return metric_values, metric_label


def plot_pareto_front_2d(
    pareto_objectives: np.ndarray,
    save_path: PathLike | None = None,
    *,
    ax: Axes | None = None,
    true_front: np.ndarray | None = None,
    title: str = "Pareto Front",
    xlabel: str = "Objective 1",
    ylabel: str = "Objective 2",
    highlight_solutions: Sequence[int] | None = None,
    show_knee_point: bool = False,
    color_by_metric: str | None = None,
    metric_values: np.ndarray | None = None,
    metric_label: str | None = None,
    dpi: int | None = None,
) -> Axes:
    """Publication-quality 2D Pareto front scatter plot.

    :param pareto_objectives: Array of shape ``(n_solutions, 2)``.
    :param save_path: If provided, save ``.pdf`` and ``.png`` to this path.
    :param ax: Optional matplotlib Axes for embedding in multi-panel figures.
    :param true_front: Optional array for a reference/true Pareto front.
    :param title: Plot title.
    :param xlabel: X-axis label (include units).
    :param ylabel: Y-axis label (include units).
    :param highlight_solutions: Indices of solutions to highlight.
    :param show_knee_point: Mark the knee (best trade-off) point.
    :param color_by_metric: ``'crowding'`` or ``'rank'`` to color points.
    :param metric_values: Custom per-solution values for coloring.
    :param metric_label: Colorbar label.
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: The matplotlib Axes used for plotting.
    """
    apply_style()
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=scale_figsize(SINGLE_COL_WIDTH, 2.8))
    else:
        fig = ax.figure

    if true_front is not None:
        ax.plot(
            true_front[:, 0],
            true_front[:, 1],
            "k--",
            linewidth=1.5,
            alpha=0.5,
            label="True Pareto Front",
            zorder=1,
        )

    metric_values, metric_label = _resolve_metric(
        color_by_metric, metric_values, metric_label, pareto_objectives
    )

    if color_by_metric and metric_values is not None:
        plot_values = _clamp_inf(metric_values)
        scatter = ax.scatter(
            pareto_objectives[:, 0],
            pareto_objectives[:, 1],
            c=plot_values,
            cmap="viridis",
            s=marker_area(1.3),
            alpha=0.88,
            edgecolors="black",
            linewidth=line_width(0.25),
            zorder=3,
        )
        cbar = fig.colorbar(scatter, ax=ax, pad=0.04)
        cbar.set_label(
            metric_label, rotation=270, labelpad=1.4 * plt.rcParams["font.size"]
        )
    else:
        ax.scatter(
            pareto_objectives[:, 0],
            pareto_objectives[:, 1],
            s=marker_area(1.3),
            alpha=0.88,
            edgecolors="black",
            linewidth=line_width(0.25),
            label="Obtained Solutions",
            zorder=3,
        )

    if show_knee_point:
        knee_idx = calculate_knee_point(pareto_objectives)
        ax.scatter(
            pareto_objectives[knee_idx, 0],
            pareto_objectives[knee_idx, 1],
            zorder=6,
            **_knee_point_style(),
        )

    if highlight_solutions is not None:
        highlighted = pareto_objectives[highlight_solutions]
        ax.scatter(
            highlighted[:, 0],
            highlighted[:, 1],
            s=marker_area(2.0),
            marker="D",
            facecolors="none",
            edgecolors="darkorange",
            linewidth=line_width(0.75),
            label="Highlighted",
            zorder=4,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    format_publication_axes(ax)
    if not color_by_metric:
        ax.legend(loc="best", frameon=True)

    finalize_figure(fig, save_path, own_fig=own_fig, dpi=dpi)

    return ax


def plot_pareto_front_3d(
    pareto_objectives: np.ndarray,
    save_path: PathLike | None = None,
    *,
    ax: Axes | None = None,
    title: str = "Pareto Front (3D)",
    labels: Sequence[str] | None = None,
    show_knee_point: bool = False,
    color_by_metric: str | None = None,
    metric_values: np.ndarray | None = None,
    metric_label: str | None = None,
    dpi: int | None = None,
) -> Axes:
    """Publication-quality 3D Pareto front scatter plot.

    :param pareto_objectives: Array of shape ``(n_solutions, 3)``.
    :param save_path: If provided, save ``.pdf`` and ``.png``.
    :param ax: Optional 3D Axes for embedding.
    :param title: Plot title.
    :param labels: List of 3 axis labels (with units).
    :param show_knee_point: Mark the knee point.
    :param color_by_metric: ``'crowding'`` or ``'rank'``.
    :param metric_values: Custom coloring values.
    :param metric_label: Colorbar label.
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: The matplotlib 3D Axes.
    """
    apply_style()
    if labels is None:
        labels = ["Objective 1", "Objective 2", "Objective 3"]
    own_fig = ax is None
    if own_fig:
        fig = plt.figure(figsize=scale_figsize(DOUBLE_COL_WIDTH, 5.2))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # Depth-based draw order would paint nearby solutions over the knee star;
    # use explicit artist zorder instead (panes/grid keep their low defaults).
    ax.computed_zorder = False

    metric_values, metric_label = _resolve_metric(
        color_by_metric, metric_values, metric_label, pareto_objectives
    )

    if color_by_metric and metric_values is not None:
        plot_values = _clamp_inf(metric_values)
        scatter = ax.scatter(
            pareto_objectives[:, 0],
            pareto_objectives[:, 1],
            pareto_objectives[:, 2],
            s=marker_area(1.2),
            alpha=0.8,
            edgecolors="black",
            linewidth=line_width(0.25),
            c=plot_values,
            cmap="viridis",
            zorder=4,
        )
        fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7, label=metric_label)
    else:
        # Color by the third objective as a depth cue: it restates the
        # z-position, which is exactly what's hard to read in a projected
        # 3D scatter, so no colorbar is needed.
        ax.scatter(
            pareto_objectives[:, 0],
            pareto_objectives[:, 1],
            pareto_objectives[:, 2],
            s=marker_area(1.2),
            alpha=0.85,
            edgecolors="black",
            linewidth=line_width(0.25),
            c=pareto_objectives[:, 2],
            cmap="viridis",
            zorder=4,
        )

    if show_knee_point:
        knee_idx = calculate_knee_point(pareto_objectives)
        ax.scatter(
            pareto_objectives[knee_idx, 0],
            pareto_objectives[knee_idx, 1],
            pareto_objectives[knee_idx, 2],
            zorder=6,
            **_knee_point_style(),
        )
        ax.legend(loc="best")

    # Stock 3D panes print as washed-out gray boxes; keep them white and zoom
    # slightly to reclaim the large default margins.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_box_aspect(None, zoom=1.12)

    label_pad = 0.8 * plt.rcParams["font.size"]
    ax.set_xlabel(labels[0], labelpad=label_pad)
    ax.set_ylabel(labels[1], labelpad=label_pad)
    ax.set_zlabel(labels[2], labelpad=label_pad)
    ax.set_title(title)

    finalize_figure(fig, save_path, own_fig=own_fig, dpi=dpi)

    return ax


def plot_pareto_front_projections(
    pareto_objectives: np.ndarray,
    save_path: PathLike | None = None,
    *,
    title: str = "Pareto Front Projections",
    objective_names: Sequence[str] | None = None,
    dpi: int | None = None,
) -> np.ndarray:
    """Pairwise 2D projections of an n-dimensional Pareto front.

    Creates a grid of scatter plots for every pair of objectives.
    Useful for fronts with 4+ objectives where 2D/3D plots are
    insufficient.

    :param pareto_objectives: Array of shape ``(n_solutions, n_objectives)``.
    :param save_path: If provided, save ``.pdf`` and ``.png``.
    :param title: Figure super-title.
    :param objective_names: Labels for each objective.
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: Array of Axes.
    """
    apply_style()
    n_obj = pareto_objectives.shape[1]
    if objective_names is None:
        objective_names = [f"Objective {i+1}" for i in range(n_obj)]

    n_pairs = n_obj * (n_obj - 1) // 2
    n_cols = min(3, n_pairs)
    n_rows = int(np.ceil(n_pairs / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=scale_figsize(SINGLE_COL_WIDTH * n_cols, 2.8 * n_rows)
    )
    if n_pairs == 1:
        axes = np.array([axes])
    else:
        axes = np.asarray(axes).flatten()

    idx = 0
    for i in range(n_obj):
        for j in range(i + 1, n_obj):
            ax = axes[idx]
            ax.scatter(
                pareto_objectives[:, i],
                pareto_objectives[:, j],
                s=marker_area(1.0),
                alpha=0.7,
                edgecolors="black",
                linewidth=line_width(0.2),
            )
            ax.set_xlabel(objective_names[i])
            ax.set_ylabel(objective_names[j])
            format_publication_axes(ax)
            idx += 1

    for k in range(idx, len(axes)):
        axes[k].axis("off")

    fig.suptitle(title)

    finalize_figure(fig, save_path, dpi=dpi)

    return axes


def _clamp_inf(values):
    """Replace infinite values with 1.2x the finite maximum for plotting."""
    out = np.copy(values).astype(float)
    if np.any(np.isinf(out)):
        finite_max = np.max(out[np.isfinite(out)])
        out[np.isinf(out)] = finite_max * 1.2
    return out
