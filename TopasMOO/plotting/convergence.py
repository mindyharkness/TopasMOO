"""
Convergence plots for optimization progress tracking.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ..io import ReadInMultiObjectiveLogFile
from .style import (
    SINGLE_COL_WIDTH,
    apply_style,
    finalize_figure,
    format_publication_axes,
    line_width,
    marker_area,
    scale_figsize,
    style_colors,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]


def _raw_evaluations(ax, iterations, values) -> None:
    """Draw the raw per-evaluation values as a faint background scatter.

    Population optimizers evaluate noisy candidates all over the search space,
    so connecting consecutive evaluations with a line yields unreadable ink;
    the trend belongs to the summary overlays drawn on top of this scatter.
    """
    ax.scatter(
        iterations,
        values,
        s=marker_area(0.55),
        color="0.55",
        alpha=0.3,
        edgecolors="none",
        zorder=1,
        rasterized=True,
    )


def _rolling_quantiles(values, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centered rolling (25th, 50th, 75th) percentiles of *values*."""
    values = np.asarray(values, dtype=float)
    half = window // 2
    q25 = np.empty_like(values)
    q50 = np.empty_like(values)
    q75 = np.empty_like(values)
    for i in range(len(values)):
        chunk = values[max(0, i - half) : i + half + 1]
        q25[i], q50[i], q75[i] = np.percentile(chunk, [25, 50, 75])
    return q25, q50, q75


def _panel_grid(n_panels, ax):
    """Create (or wrap) the up-to-2-column panel grid both convergence plots use.

    :param n_panels: Number of panels that will be drawn.
    :param ax: Caller-supplied Axes to embed into, or ``None`` to create a figure.

    :returns: Tuple ``(fig, axes, axes_flat, own_fig)`` where ``axes`` preserves the
        shape callers return and ``axes_flat`` is a flat list with any surplus
        grid cells already hidden.
    """
    n_cols = min(n_panels, 2)
    n_rows = (n_panels + n_cols - 1) // n_cols
    own_fig = ax is None
    if own_fig:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=scale_figsize(SINGLE_COL_WIDTH * n_cols, 2.65 * n_rows),
            sharex=True,
        )
    else:
        fig = ax.figure
        axes = ax

    if isinstance(axes, np.ndarray):
        axes_flat = list(axes.flatten())
    else:
        axes_flat = [axes]
    for surplus in axes_flat[n_panels:]:
        surplus.set_visible(False)
    return fig, axes, axes_flat, own_fig


def plot_objective_convergence(
    log_file_path: PathLike,
    save_path: PathLike | None = None,
    *,
    n_objectives: int | None = None,
    ax: Axes | None = None,
    dpi: int | None = None,
) -> Axes | np.ndarray | None:
    """Plot convergence of each objective function over iterations.

    :param log_file_path: Path to the optimization log file.
    :param save_path: If provided, save ``.pdf`` and ``.png``.
    :param n_objectives: Number of objectives (inferred from log if None).
    :param ax: Optional Axes (only used when ``n_objectives == 1``).
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: Array of Axes (or single Axes if ``n_objectives == 1``).
    """
    apply_style()
    results = ReadInMultiObjectiveLogFile(log_file_path)

    if "Iteration" not in results:
        logger.warning("Could not find iteration data in log file")
        return None

    iterations = results["Iteration"]

    if n_objectives is None:
        n_objectives = sum(1 for k in results if k.startswith("ObjectiveFunction_"))

    fig, axes, axes_flat, own_fig = _panel_grid(n_objectives, ax)

    accent = style_colors(1)[0]
    for i in range(n_objectives):
        key = f"ObjectiveFunction_{i + 1}"
        if key in results and i < len(axes_flat):
            panel_ax = axes_flat[i]
            values = np.asarray(results[key], dtype=float)
            _raw_evaluations(panel_ax, iterations, values)
            panel_ax.plot(
                iterations,
                np.minimum.accumulate(values),
                drawstyle="steps-post",
                linewidth=line_width(1.0),
                color=accent,
                zorder=3,
                label="Running best" if i == 0 else None,
            )
            panel_ax.set_xlabel("Iteration")
            panel_ax.set_ylabel(f"Objective {i + 1}")
            panel_ax.set_title(f"Convergence: Objective {i + 1}")
            format_publication_axes(panel_ax, x_integer=True)
            if i == 0:
                panel_ax.legend(loc="upper right")

    finalize_figure(fig, save_path, own_fig=own_fig, dpi=dpi)

    return axes


def plot_parameter_convergence(
    log_file_path: PathLike,
    save_path: PathLike | None = None,
    parameter_names: Sequence[str] | None = None,
    *,
    ax: Axes | None = None,
    dpi: int | None = None,
) -> Axes | np.ndarray | None:
    """Plot convergence of optimization parameters over iterations.

    :param log_file_path: Path to the optimization log file.
    :param save_path: If provided, save ``.pdf`` and ``.png``.
    :param parameter_names: List of parameter names to plot.
    :param ax: Optional Axes (only for single parameter).
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: Array of Axes.
    """
    apply_style()
    results = ReadInMultiObjectiveLogFile(log_file_path)

    if "Iteration" not in results:
        logger.warning("Could not find iteration data in log file")
        return None

    if parameter_names is None:
        parameter_names = [
            k
            for k in results
            if k not in ("Iteration",) and not k.startswith("ObjectiveFunction_")
        ]

    iterations = results["Iteration"]
    n_params = len(parameter_names)
    fig, axes, axes_flat, own_fig = _panel_grid(n_params, ax)

    accent = style_colors(1)[0]
    for i, name in enumerate(parameter_names):
        if name in results and i < len(axes_flat):
            panel_ax = axes_flat[i]
            values = np.asarray(results[name], dtype=float)
            _raw_evaluations(panel_ax, iterations, values)
            # Parameters have no "best" direction; show where the population
            # concentrates instead: rolling median with an interquartile band.
            window = max(5, len(values) // 25)
            q25, q50, q75 = _rolling_quantiles(values, window)
            panel_ax.fill_between(
                iterations, q25, q75, color=accent, alpha=0.18,
                linewidth=0, zorder=2,
                label="Interquartile range" if i == 0 else None,
            )
            panel_ax.plot(
                iterations, q50, linewidth=line_width(1.0), color=accent,
                zorder=3, label="Rolling median" if i == 0 else None,
            )
            panel_ax.set_xlabel("Iteration")
            panel_ax.set_ylabel(name)
            panel_ax.set_title(f"Parameter: {name}")
            format_publication_axes(panel_ax, x_integer=True)
            if i == 0:
                panel_ax.legend(loc="upper right")

    finalize_figure(fig, save_path, own_fig=own_fig, dpi=dpi)

    return axes
