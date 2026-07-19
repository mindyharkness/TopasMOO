"""
Hypervolume convergence plot.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .style import (
    SINGLE_COL_WIDTH,
    apply_style,
    finalize_figure,
    format_publication_axes,
    line_width,
    scale_figsize,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]


def plot_hypervolume_convergence(
    hv_history: Sequence[float] | np.ndarray,
    save_path: PathLike | None = None,
    *,
    ax: Axes | None = None,
    title: str = "Hypervolume Convergence",
    xlabel: str = "Generation",
    ylabel: str = "Hypervolume",
    reference_hv: float | None = None,
    dpi: int | None = None,
) -> Axes:
    """Plot hypervolume indicator vs. generation.

    :param hv_history: Sequence of hypervolume values, one per generation.
    :param save_path: If provided, save ``.pdf`` and ``.png`` to this path.
    :param ax: Optional matplotlib Axes for embedding in multi-panel figures.
    :param title: Plot title.
    :param xlabel: X-axis label.
    :param ylabel: Y-axis label.
    :param reference_hv: If given, draw a horizontal dashed reference line
        (e.g. theoretical maximum for a benchmark problem).
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: The matplotlib Axes used for plotting.
    """
    apply_style()
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=scale_figsize(SINGLE_COL_WIDTH, 2.8))
    else:
        fig = ax.figure

    hv = np.asarray(hv_history, dtype=float)
    generations = np.arange(len(hv))

    ax.plot(generations, hv, linewidth=line_width(0.9), zorder=3)
    ax.fill_between(generations, hv, alpha=0.12, zorder=2)

    if reference_hv is not None:
        ax.axhline(
            reference_hv,
            linestyle="--",
            linewidth=line_width(0.55),
            alpha=0.7,
            color="#555555",
            label=f"Reference ({reference_hv:.4g})",
            zorder=1,
        )
        ax.legend(loc="lower right")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(left=0)
    format_publication_axes(ax, x_integer=True)

    finalize_figure(fig, save_path, own_fig=own_fig, dpi=dpi)

    return ax
