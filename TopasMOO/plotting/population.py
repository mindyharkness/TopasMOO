"""Population evolution plot -- shows convergence across generations."""

from __future__ import annotations

import logging
import os
from typing import Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

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
PopulationSnapshot = Tuple[int, np.ndarray]

_N_SNAPSHOTS = 5  # how many generation snapshots to overlay


def plot_population_evolution(
    population_history: Sequence[PopulationSnapshot],
    save_path: PathLike | None = None,
    *,
    ax: Axes | None = None,
    title: str = "Population Evolution",
    xlabel: str = "Objective 1",
    ylabel: str = "Objective 2",
    n_snapshots: int = _N_SNAPSHOTS,
    dpi: int | None = None,
) -> Axes | None:
    """Overlay population snapshots to show convergence toward the Pareto front.

    Selects up to *n_snapshots* evenly spaced generations and draws them as
    scatter plots from light to dark using a Blues color ramp.

    :param population_history: List of ``(generation_index, objectives_array)``
        tuples as stored in ``optimizer.PopulationHistory``.
    :param save_path: If provided, save ``.pdf`` and ``.png`` to this path.
    :param ax: Optional matplotlib Axes for embedding in multi-panel figures.
    :param title: Plot title.
    :param xlabel: X-axis label (first objective).
    :param ylabel: Y-axis label (second objective).
    :param n_snapshots: Number of generation snapshots to draw (max).
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: The matplotlib Axes used for plotting.
    """
    if not population_history:
        logger.warning("population_history is empty; skipping population evolution plot.")
        return None

    apply_style()
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=scale_figsize(SINGLE_COL_WIDTH, 2.8))
    else:
        fig = ax.figure

    n_gens = len(population_history)
    n_snap = min(n_snapshots, n_gens)
    indices = np.round(np.linspace(0, n_gens - 1, n_snap)).astype(int)

    colors = plt.cm.Blues(np.linspace(0.3, 0.95, n_snap))

    for plot_idx, gen_idx in enumerate(indices):
        gen_num, objectives = population_history[gen_idx]
        is_last = plot_idx == len(indices) - 1
        frac = plot_idx / max(len(indices) - 1, 1)
        alpha = 0.35 + 0.55 * frac
        s = marker_area(0.75 + 0.45 * frac)
        ax.scatter(
            objectives[:, 0],
            objectives[:, 1],
            s=s,
            c=[colors[plot_idx]],
            alpha=alpha,
            edgecolors="none" if not is_last else "#333333",
            linewidth=0 if not is_last else line_width(0.2),
            label=f"Gen {gen_num}" if (plot_idx == 0 or is_last) else "_nolegend_",
            zorder=2 + plot_idx,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    format_publication_axes(ax)
    ax.legend(loc="upper right")

    finalize_figure(fig, save_path, own_fig=own_fig, dpi=dpi)

    return ax
