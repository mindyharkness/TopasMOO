"""Nightingale rose (petal) diagrams for solution comparison."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..metrics import normalize_objectives
from .style import (
    SINGLE_COL_WIDTH,
    apply_style,
    finalize_figure,
    line_width,
    scale_figsize,
    style_colors,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]


def _draw_petals(
    ax,
    normalized,
    labels,
    *,
    edge_factor: float,
    ytick_vals,
    ytick_size: str,
    label_size: str | None = None,
):
    """Draw one solution's petals on a polar Axes (shared single/multi core).

    :param ax: Polar Axes to draw into.
    :param normalized: Per-objective petal lengths in ``[0, 1]``.
    :param labels: One label per petal.
    :param edge_factor: Petal edge width as a :func:`line_width` factor.
    :param ytick_vals: Radial tick positions (also used as tick labels).
    :param ytick_size: Font size for the radial tick labels.
    :param label_size: Font size for the petal labels (style default when None).
    """
    n_obj = len(normalized)
    theta = np.linspace(0, 2 * np.pi, n_obj, endpoint=False)
    width = 2 * np.pi / n_obj

    bars = ax.bar(
        theta,
        normalized,
        width=width,
        bottom=0.0,
        alpha=0.8,
        edgecolor="black",
        linewidth=line_width(edge_factor),
    )
    for bar, color in zip(bars, style_colors(n_obj), strict=False):
        bar.set_facecolor(color)

    ax.set_xticks(theta)
    label_kwargs = {} if label_size is None else {"fontsize": label_size}
    ax.set_xticklabels(labels, **label_kwargs)
    ax.set_ylim(0, 1.1)
    ax.set_yticks(ytick_vals)
    ax.set_yticklabels([str(v) for v in ytick_vals], fontsize=ytick_size)
    ax.grid(True)


def plot_petal_diagram_single(
    solution_objectives: np.ndarray,
    save_path: PathLike | None = None,
    *,
    ax: Axes | None = None,
    solution_label: str = "Solution",
    objective_names: Sequence[str] | None = None,
    ideal: np.ndarray | None = None,
    nadir: np.ndarray | None = None,
    dpi: int | None = None,
) -> Axes | None:
    """Nightingale rose chart for a single multi-objective solution.

    Each petal represents one objective.  Larger petals indicate better
    performance, so all objectives are assumed to be minimized.

    Normalization:
        When ``ideal`` and ``nadir`` are supplied, each objective is normalized
        per-objective against those shared bounds, i.e.
        ``1 - (obj - ideal) / (nadir - ideal)``.  This keeps petals comparable
        across solutions that share a scale (use this when comparing multiple
        solutions, e.g. via :func:`plot_petal_diagram_multi`).  When the bounds
        are omitted, the single solution is normalized against the largest of
        its own objective magnitudes; this is a self-relative view only and
        should not be compared across solutions or objectives with different
        scales.

    :param solution_objectives: 1-D array of objective values.
    :param save_path: If provided, save ``.pdf`` and ``.png``.
    :param ax: Optional polar Axes for embedding.
    :param solution_label: Title for this solution.
    :param objective_names: Labels for each petal.
    :param ideal: Optional per-objective ideal (minimum) values for shared
        normalization.  Must be given together with ``nadir``.
    :param nadir: Optional per-objective nadir (maximum) values for shared
        normalization.  Must be given together with ``ideal``.
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: The polar Axes, or ``None`` when fewer than 3 objectives are given
        (a 2-objective petal is just two semicircles; use the Pareto plots).
    """
    solution_objectives = np.asarray(solution_objectives, dtype=float)
    n_obj = len(solution_objectives)
    if n_obj < 3:
        logger.warning(
            "Petal diagrams need at least 3 objectives to be meaningful "
            "(got %d); skipping. Use plot_pareto_front_2d instead.", n_obj,
        )
        return None
    apply_style()

    if ideal is not None and nadir is not None:
        ideal = np.asarray(ideal, dtype=float)
        nadir = np.asarray(nadir, dtype=float)
        obj_range = nadir - ideal
        obj_range[obj_range == 0] = 1.0
        # Floor petals at a sliver so even the worst objective stays visible.
        normalized = np.clip(1.0 - (solution_objectives - ideal) / obj_range, 0.03, 1.0)
    else:
        max_val = np.max(np.abs(solution_objectives))
        if max_val == 0:
            max_val = 1
        normalized = np.clip(1.0 - solution_objectives / max_val, 0.03, 1.0)

    own_fig = ax is None
    if own_fig:
        fig = plt.figure(figsize=scale_figsize(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH))
        ax = fig.add_subplot(111, projection="polar")
    else:
        fig = ax.figure

    labels = objective_names or [f"Obj {i+1}" for i in range(n_obj)]
    _draw_petals(
        ax,
        normalized,
        labels,
        edge_factor=0.5,
        ytick_vals=[0.25, 0.5, 0.75, 1.0],
        ytick_size="x-small",
    )
    ax.set_title(solution_label, pad=15)

    finalize_figure(fig, save_path, own_fig=own_fig, dpi=dpi)

    return ax


def plot_petal_diagram_multi(
    pareto_objectives: np.ndarray,
    save_dir: PathLike | None = None,
    *,
    title: str = "Solution Comparison (Petal Diagrams)",
    objective_names: Sequence[str] | None = None,
    max_solutions: int = 9,
    dpi: int | None = None,
) -> Figure | None:
    """Multi-panel nightingale rose charts comparing Pareto solutions.

    Solutions are normalized together (global ideal/nadir) so that
    petal sizes are comparable across panels.

    :param pareto_objectives: Array of shape ``(n_solutions, n_objectives)``.
    :param save_dir: Directory for individual and multi-panel figures.
    :param title: Figure super-title.
    :param objective_names: Labels for each objective.
    :param max_solutions: Maximum panels to plot (evenly sampled).
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: The matplotlib Figure, or ``None`` when fewer than 3 objectives are
        given.
    """
    n_solutions, n_obj = pareto_objectives.shape
    if n_obj < 3:
        logger.warning(
            "Petal diagrams need at least 3 objectives to be meaningful "
            "(got %d); skipping. Use plot_pareto_front_2d instead.", n_obj,
        )
        return None
    apply_style()

    if n_solutions > max_solutions:
        indices = np.linspace(0, n_solutions - 1, max_solutions, dtype=int)
        plot_solutions = pareto_objectives[indices]
        solution_indices = indices
    else:
        plot_solutions = pareto_objectives
        solution_indices = np.arange(n_solutions)

    n_plot = len(plot_solutions)

    # Shared ideal/nadir so every panel (and the individual files) normalize
    # each objective on the same scale and stay comparable across solutions.
    # ``normalized_all`` holds (obj - ideal) / range per solution; each petal
    # shows 1 - that value so larger petals mean better (smaller) objectives.
    normalized_all, ideal, nadir = normalize_objectives(pareto_objectives)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for sol_obj, sol_idx in zip(plot_solutions, solution_indices, strict=False):
            plot_petal_diagram_single(
                sol_obj,
                save_dir / f"petal_solution_{sol_idx + 1}",
                solution_label=f"Solution {sol_idx + 1}",
                objective_names=objective_names,
                ideal=ideal,
                nadir=nadir,
                dpi=dpi,
            )

    n_cols = int(np.ceil(np.sqrt(n_plot)))
    n_rows = int(np.ceil(n_plot / n_cols))

    panel = 0.9 * SINGLE_COL_WIDTH
    fig = plt.figure(figsize=scale_figsize(panel * n_cols, panel * n_rows))

    for i, sol_idx in enumerate(solution_indices):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="polar")
        normalized = np.clip(1.0 - normalized_all[sol_idx], 0.03, 1.0)
        labels = objective_names or [f"O{j+1}" for j in range(n_obj)]
        _draw_petals(
            ax,
            normalized,
            labels,
            edge_factor=0.4,
            ytick_vals=[0.5, 1.0],
            ytick_size="xx-small",
            label_size="x-small",
        )
        ax.set_title(f"Sol {sol_idx + 1}", fontsize="small")

    fig.suptitle(title)

    finalize_figure(
        fig,
        (save_dir / "petal_diagram_multipanel") if save_dir is not None else None,
        dpi=dpi,
    )

    return fig
