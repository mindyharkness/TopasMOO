"""Parallel coordinates plot for many-objective Pareto fronts."""

from __future__ import annotations

import logging
import os
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.axes import Axes

from ..metrics import calculate_knee_point
from .style import (
    DOUBLE_COL_WIDTH,
    apply_style,
    finalize_figure,
    format_publication_axes,
    line_width,
    scale_figsize,
    style_colors,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]


def plot_parallel_coordinates(
    pareto_objectives: np.ndarray,
    save_path: PathLike | None = None,
    *,
    ax: Axes | None = None,
    dominance_rank: np.ndarray | None = None,
    highlight_solutions: Sequence[int] | None = None,
    highlight_knee: bool = True,
    highlight_extremes: bool = False,
    title: str = "Parallel Coordinates",
    objective_names: Sequence[str] | None = None,
    dpi: int | None = None,
) -> Axes:
    """Parallel coordinates plot for n-dimensional objective values.

    Each solution is drawn as a polyline connecting its normalized objective
    values across parallel vertical axes — useful for reading trade-offs when
    there are more than three objectives. To keep the plot legible the bulk of
    the front is drawn as faint background lines and a few reference solutions
    are picked out on top:

    * the knee point (best balanced trade-off), when ``highlight_knee`` is set;
    * the per-objective best solutions, when ``highlight_extremes`` is set;
    * any explicit ``highlight_solutions`` the caller passes.

    Passing ``dominance_rank`` instead colors every line by rank (with a
    colorbar) and disables the automatic knee/extreme highlighting.

    :param pareto_objectives: Array of shape ``(n_solutions, n_objectives)``.
    :param save_path: If provided, save ``.pdf`` and ``.png``.
    :param ax: Optional Axes for embedding.
    :param dominance_rank: Per-solution rank array for a color gradient.
    :param highlight_solutions: Indices of solutions to draw prominently.
    :param highlight_knee: Auto-highlight the knee-point solution (ignored when
        ``dominance_rank`` or ``highlight_solutions`` is given).
    :param highlight_extremes: Also highlight the best solution per objective.
    :param title: Plot title.
    :param objective_names: Labels for each vertical axis.
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: The matplotlib Axes.
    """
    apply_style()
    n_solutions, n_objectives = pareto_objectives.shape

    obj_min = pareto_objectives.min(axis=0)
    obj_max = pareto_objectives.max(axis=0)
    obj_range = obj_max - obj_min
    obj_range[obj_range == 0] = 1
    normalized = (pareto_objectives - obj_min) / obj_range

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=scale_figsize(DOUBLE_COL_WIDTH, 3.8))
    else:
        fig = ax.figure

    x_pos = np.arange(n_objectives)

    # Axis grid lines for each objective sit behind everything else.
    for xp in x_pos:
        ax.axvline(x=xp, color="0.55", linewidth=line_width(0.5), zorder=0)

    # --- Background lines ----------------------------------------------------
    norm = None
    if dominance_rank is not None:
        norm = plt.Normalize(vmin=dominance_rank.min(), vmax=dominance_rank.max())
        for i in range(n_solutions):
            ax.plot(
                x_pos, normalized[i], color=cm.viridis(norm(dominance_rank[i])),
                alpha=0.7, linewidth=line_width(0.7), zorder=2,
            )
    else:
        for i in range(n_solutions):
            ax.plot(x_pos, normalized[i], color="0.6", alpha=0.25, linewidth=line_width(0.55), zorder=1)

    # --- Highlighted reference solutions -------------------------------------
    # Explicit highlight_solutions are always honored (even alongside a
    # dominance-rank gradient); knee/extremes are only auto-picked when the
    # caller hasn't supplied explicit highlights or a rank gradient.
    highlights: list[tuple[int, str]] = []  # (solution index, legend label)
    if highlight_solutions is not None:
        highlights = [(int(i), "Highlighted") for i in highlight_solutions]
    elif dominance_rank is None:
        if highlight_knee and n_solutions > 1:
            highlights.append((int(calculate_knee_point(pareto_objectives)), "Knee point"))
        if highlight_extremes:
            for obj in range(n_objectives):
                best = int(np.argmin(pareto_objectives[:, obj]))
                label = f"Best {objective_names[obj]}" if objective_names else f"Best obj {obj + 1}"
                highlights.append((best, label))

    accent = style_colors(max(len(highlights), 1))
    seen_labels: set[str] = set()
    for (idx, label), color in zip(highlights, accent, strict=False):
        ax.plot(
            x_pos, normalized[idx], color=color, alpha=0.95, linewidth=line_width(1.5),
            zorder=4, label=label if label not in seen_labels else None,
            marker="o", markersize=1.1 * plt.rcParams["lines.markersize"],
        )
        seen_labels.add(label)

    ax.set_xticks(x_pos)
    if objective_names is not None:
        ax.set_xticklabels(objective_names)
    else:
        ax.set_xticklabels([f"Obj {i+1}" for i in range(n_objectives)])

    ax.set_ylabel("Normalized Value")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    format_publication_axes(ax)

    if norm is not None:
        sm = cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.04)
        cbar.set_label(
            "Dominance Rank", rotation=270, labelpad=1.4 * plt.rcParams["font.size"]
        )
    # A rank gradient and explicit highlights can coexist, so show the legend
    # whenever any line is labeled — independently of the colorbar.
    if seen_labels:
        ax.legend(loc="best")

    finalize_figure(fig, save_path, own_fig=own_fig, dpi=dpi)

    return ax
