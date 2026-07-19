"""
Decision variable heatmap for the Pareto set.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from .style import (
    ACCENT_COLOR,
    DOUBLE_COL_WIDTH,
    apply_style,
    finalize_figure,
    format_publication_axes,
    line_width,
    scale_figsize,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]


def plot_decision_heatmap(
    decision_vars: np.ndarray,
    save_path: PathLike | None = None,
    *,
    parameter_names: Sequence[str] | None = None,
    title: str = "Decision Variables (Pareto Set)",
    dpi: int | None = None,
) -> tuple[Axes, Axes]:
    """Two-panel figure: normalized heatmap of Pareto decision variables + boxplots.

    The left panel shows an imshow heatmap (solutions × parameters) sorted by
    the first parameter.  The right panel shows per-parameter box distributions
    so the spread and central tendency of each variable are immediately visible.
    Both panels use the same per-parameter [0, 1] normalization; raw-value
    boxplots would flatline any parameter whose scale is much smaller than the
    others'.

    :param decision_vars: Array of shape ``(n_solutions, n_params)``.
    :param save_path: If provided, save ``.pdf`` and ``.png`` to this path.
    :param parameter_names: List of parameter name strings.  Falls back to
        ``["p1", "p2", ...]`` if not given.
    :param title: Figure suptitle.
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: Tuple of ``(ax_heat, ax_box)`` axes.
    """
    apply_style()

    decision_vars = np.asarray(decision_vars, dtype=float)
    n_solutions, n_params = decision_vars.shape

    if parameter_names is None:
        parameter_names = [f"p{i + 1}" for i in range(n_params)]

    # Normalize each column to [0, 1] for the heatmap
    col_min = decision_vars.min(axis=0)
    col_max = decision_vars.max(axis=0)
    col_range = np.where(col_max - col_min > 0, col_max - col_min, 1.0)
    normalized = (decision_vars - col_min) / col_range

    # Sort rows by first parameter for visual clarity
    order = np.argsort(decision_vars[:, 0])
    normalized_sorted = normalized[order]

    fig, (ax_heat, ax_box) = plt.subplots(
        1,
        2,
        figsize=scale_figsize(DOUBLE_COL_WIDTH, max(3.2, 0.42 * n_params + 2.2)),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    # Left: heatmap
    im = ax_heat.imshow(
        normalized_sorted.T,
        cmap="viridis",
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax_heat.set_xlabel("Solution index (sorted by first parameter)")
    ax_heat.set_ylabel("Parameter")
    ax_heat.set_title("Normalized values")
    ax_heat.set_yticks(range(n_params))
    ax_heat.set_yticklabels(parameter_names)
    cbar = fig.colorbar(im, ax=ax_heat, pad=0.02, fraction=0.08)
    cbar.set_label("Normalized value")

    # Right: boxplots on the same normalized scale as the heatmap
    positions = np.arange(1, n_params + 1)
    bp = ax_box.boxplot(
        [normalized[:, i] for i in range(n_params)],
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        medianprops=dict(linewidth=line_width(0.75), color="#333333"),
        meanprops=dict(linewidth=line_width(0.75), linestyle="--", color=ACCENT_COLOR),
        whiskerprops=dict(linewidth=line_width(0.5)),
        capprops=dict(linewidth=line_width(0.5)),
        flierprops=dict(
            marker="o", markersize=0.65 * plt.rcParams["lines.markersize"], alpha=0.5
        ),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#A8DADC")
        patch.set_alpha(0.7)

    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(parameter_names, rotation=45, ha="right")
    ax_box.set_ylabel("Normalized value")
    ax_box.set_title("Distribution")
    ax_box.set_ylim(-0.05, 1.05)
    format_publication_axes(ax_box)

    # Legend for mean vs median
    legend_elements = [
        Line2D([0], [0], color="#333333", linewidth=line_width(0.75), label="Median"),
        Line2D(
            [0], [0], color=ACCENT_COLOR, linewidth=line_width(0.75), linestyle="--",
            label="Mean",
        ),
    ]
    ax_box.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.04, 0.5),
        frameon=False,
        borderaxespad=0,
    )

    fig.suptitle(title)

    finalize_figure(fig, save_path, dpi=dpi)

    return ax_heat, ax_box
