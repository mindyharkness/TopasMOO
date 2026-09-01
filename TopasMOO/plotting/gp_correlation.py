"""Gaussian-process prediction correlation plotting for MOBO runs."""

from __future__ import annotations

import logging
import os
from math import ceil
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

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


def _correlation_coefficients(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float]:
    """Return ``(Pearson, Spearman)`` coefficients, or NaNs when undefined."""
    if len(observed) < 3 or np.ptp(observed) == 0 or np.ptp(predicted) == 0:
        return float("nan"), float("nan")
    return (
        float(pearsonr(observed, predicted).statistic),
        float(spearmanr(observed, predicted).statistic),
    )


def _equal_axis_limits(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """Return padded limits shared by the observed and predicted axes."""
    values = np.concatenate([observed, predicted])
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    if span == 0:
        span = max(abs(lower) * 0.1, 1.0)
    margin = 0.05 * span
    return lower - margin, upper + margin


def plot_gp_prediction_correlation(
    observed_objectives: np.ndarray,
    predicted_objectives: np.ndarray,
    save_path: PathLike | None = None,
    *,
    objective_names: Sequence[str] | None = None,
    valid_mask: np.ndarray | None = None,
    title: str = "GP Prediction Correlation",
    dpi: int | None = None,
) -> np.ndarray:
    """Plot prospective GP predictions against observed objective values.

    One panel is produced per objective. Each panel uses only rows selected by
    ``valid_mask`` whose observed and predicted values are both finite. This
    naturally excludes an initial MOBO design, for which no pre-evaluation GP
    prediction exists and prediction history is recorded as ``NaN``.

    :param observed_objectives: Observed minimization objectives with shape
        ``(n_observations, n_objectives)``.
    :param predicted_objectives: GP posterior means recorded before evaluating
        the corresponding designs, with the same shape as
        ``observed_objectives`` and in minimization space.
    :param save_path: If provided, save ``.pdf`` and ``.png``.
    :param objective_names: Optional label for each objective.
    :param valid_mask: Optional boolean mask selecting observations to include,
        for example to exclude penalized evaluation failures.
    :param title: Figure super-title.
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: A 2-D array of matplotlib Axes. Unused trailing panels are hidden.
    """
    observed = np.asarray(observed_objectives, dtype=float)
    predicted = np.asarray(predicted_objectives, dtype=float)
    if observed.ndim != 2 or observed.shape[1] < 1:
        raise ValueError(
            "observed_objectives must have shape "
            f"(n_observations, n_objectives); got {observed.shape}."
        )
    if predicted.shape != observed.shape:
        raise ValueError(
            "predicted_objectives must have the same shape as "
            f"observed_objectives; got {predicted.shape} and {observed.shape}."
        )

    n_observations, n_objectives = observed.shape
    if objective_names is not None and len(objective_names) != n_objectives:
        raise ValueError(
            "objective_names must contain one label per objective; "
            f"got {len(objective_names)} labels for {n_objectives} objectives."
        )
    if valid_mask is None:
        selected = np.ones(n_observations, dtype=bool)
    else:
        selected = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if selected.shape != (n_observations,):
            raise ValueError(
                "valid_mask must contain one entry per observation; "
                f"got shape {np.shape(valid_mask)} for {n_observations} observations."
            )

    apply_style()
    n_cols = min(3, n_objectives)
    n_rows = ceil(n_objectives / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=scale_figsize(0.95 * SINGLE_COL_WIDTH * n_cols, 3.0 * n_rows),
        squeeze=False,
    )

    for objective_index, ax in enumerate(axes.flat):
        if objective_index >= n_objectives:
            ax.set_visible(False)
            continue

        finite = (
            selected
            & np.isfinite(observed[:, objective_index])
            & np.isfinite(predicted[:, objective_index])
        )
        actual = observed[finite, objective_index]
        expected = predicted[finite, objective_index]

        if len(actual):
            limits = _equal_axis_limits(actual, expected)
            ax.scatter(
                actual,
                expected,
                s=marker_area(1.0),
                alpha=0.75,
                edgecolors="black",
                linewidth=line_width(0.2),
                zorder=3,
            )
            ax.plot(
                limits,
                limits,
                linestyle="--",
                color=ACCENT_COLOR,
                linewidth=line_width(0.65),
                label="Perfect prediction",
                zorder=2,
            )
            ax.set_xlim(limits)
            ax.set_ylim(limits)

        pearson, spearman = _correlation_coefficients(actual, expected)
        if np.isfinite(pearson) and np.isfinite(spearman):
            metrics = (
                f"n = {len(actual)}\n"
                f"Spearman: {spearman:.2f}\n"
                f"Pearson: {pearson:.2f}"
            )
        else:
            metrics = f"n = {len(actual)}\nSpearman: n/a\nPearson: n/a"
        ax.text(
            0.04,
            0.96,
            metrics,
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.85},
        )

        objective_label = (
            objective_names[objective_index]
            if objective_names is not None
            else f"Objective {objective_index + 1}"
        )
        ax.set_xlabel(f"Observed {objective_label}")
        ax.set_ylabel(f"Predicted {objective_label}")
        ax.set_title(objective_label)
        format_publication_axes(ax)

    fig.suptitle(title)
    finalize_figure(fig, save_path, dpi=dpi)
    return axes
