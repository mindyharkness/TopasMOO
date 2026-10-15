"""Run progress: hypervolume vs evaluations, constraint check, and run monitor."""

from __future__ import annotations

import logging
import os
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter
from pymoo.indicators.hv import HV

from ..metrics import calculate_dominance_rank, hypervolume_reference_point
from .pareto import plot_pareto_front_2d
from .style import (
    DOUBLE_COL_WIDTH,
    SINGLE_COL_WIDTH,
    apply_style,
    finalize_figure,
    format_publication_axes,
    scale_figsize,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]


def _as_runs(objectives, eligible):
    """Normalize one run or a list of runs to lists of arrays and masks."""
    if isinstance(objectives, np.ndarray) and objectives.ndim == 2:
        objectives, eligible = [objectives], [eligible]
    eligible = eligible if eligible is not None else [None] * len(objectives)
    runs = [np.asarray(F, dtype=float) for F in objectives]
    masks = [np.ones(len(F), bool) if m is None else np.asarray(m, bool) for F, m in zip(runs, eligible, strict=True)]
    return runs, masks


def nondominated_eligible(objectives: np.ndarray, eligible: np.ndarray | None = None) -> np.ndarray:
    """Indices of the non-dominated designs among the eligible ones."""
    objectives = np.asarray(objectives, dtype=float)
    rows = np.flatnonzero(np.ones(len(objectives), bool) if eligible is None else eligible)
    if not len(rows):
        return rows
    return rows[calculate_dominance_rank(objectives[rows]) == 0]


def hypervolume_curve(
    objectives: np.ndarray,
    ref_point: np.ndarray,
    eligible: np.ndarray | None = None,
    n_points: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Hypervolume of the eligible designs found after each number of evaluations.

    The front is grown incrementally between checkpoints, so this stays cheap
    for long runs. Designs outside the reference box add nothing.

    :returns: ``(evaluations, hypervolume)`` arrays.
    """
    objectives = np.asarray(objectives, dtype=float)
    ok = np.ones(len(objectives), bool) if eligible is None else np.asarray(eligible, bool)
    n = len(objectives)
    if n == 0:
        return np.zeros(0, int), np.zeros(0)
    checkpoints = np.unique(np.linspace(1, n, min(n, n_points)).astype(int))
    indicator = HV(ref_point=np.asarray(ref_point, dtype=float))
    front, start, hv = np.empty((0, objectives.shape[1])), 0, []
    for k in checkpoints:
        new = objectives[start:k][ok[start:k]]
        new = new[np.all(new < ref_point, axis=1)]
        if len(new):
            candidates = np.vstack([front, new])
            front = candidates[calculate_dominance_rank(candidates) == 0]
        hv.append(float(indicator(front)) if len(front) else 0.0)
        start = k
    return checkpoints, np.asarray(hv)


def plot_hypervolume_vs_evaluations(
    objectives: np.ndarray | Sequence[np.ndarray],
    save_path: PathLike | None = None,
    *,
    eligible: np.ndarray | Sequence[np.ndarray] | None = None,
    labels: Sequence[str] | None = None,
    ref_point: Sequence[float] | None = None,
    log_x: bool = True,
    ax: Axes | None = None,
    dpi: int | None = None,
) -> Axes:
    """Hypervolume after each number of evaluations, one line per run.

    All runs share one reference point (by default derived from every eligible
    evaluation of every run), so their curves can be compared directly. The
    x-axis counts evaluations, i.e. TOPAS runs, so algorithms with different
    population or batch sizes line up.

    :param objectives: ``(n_evaluations, n_objectives)`` array in evaluation
        order, minimized, or a list of such arrays (one per run).
    :param save_path: If provided, save ``.pdf`` and ``.png``.
    :param eligible: Matching mask(s) of feasible, successful evaluations.
    :param labels: One legend label per run (no legend for a single run).
    :param ref_point: Hypervolume reference point, in minimized objective space.
    :param log_x: Logarithmic evaluation axis (useful when budgets differ).
    :param ax: Optional Axes to draw into (the figure is then not saved).
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: The matplotlib Axes.
    """
    runs, masks = _as_runs(objectives, eligible)
    if ref_point is None:
        pooled = np.vstack([F[m] for F, m in zip(runs, masks, strict=True)] or [np.empty((0, 1))])
        ref_point = hypervolume_reference_point(pooled) if len(pooled) else None
    apply_style()
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=scale_figsize(SINGLE_COL_WIDTH, 2.8))
    else:
        fig = ax.figure
    for i, (F, m) in enumerate(zip(runs, masks, strict=True)):
        if ref_point is None:
            break
        evaluations, hv = hypervolume_curve(F, np.asarray(ref_point, dtype=float), m)
        ax.plot(evaluations, hv, label=None if labels is None else labels[i])
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Hypervolume")
    format_publication_axes(ax, x_integer=True)
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
    if labels is not None:
        ax.legend(loc="best")
    finalize_figure(fig, save_path, own_fig=own_fig, dpi=dpi)
    return ax


def plot_constraint_check(
    eligible: np.ndarray,
    save_path: PathLike | None = None,
    *,
    window: int | None = None,
    ax: Axes | None = None,
    dpi: int | None = None,
) -> Axes:
    """Share of feasible designs over a moving window of evaluations.

    A rising line means the optimizer has learned where the constraints are
    satisfied. Failed evaluations count as not feasible.

    :param eligible: Boolean mask of feasible, successful evaluations, in order.
    :param save_path: If provided, save ``.pdf`` and ``.png``.
    :param window: Window length; defaults to 5 % of the run, at least 20.
    :param ax: Optional Axes to draw into (the figure is then not saved).
    :param dpi: Raster resolution (defaults to the publication DPI floor).

    :returns: The matplotlib Axes.
    """
    ok = np.asarray(eligible, dtype=float)
    window = min(window or max(20, len(ok) // 20), max(len(ok), 1))
    # Sum the 0/1 counts, then divide once: exact on every platform, where
    # summing window copies of 1/window can land on 1.0000000000000002.
    share = np.convolve(ok, np.ones(window), mode="valid") / window
    apply_style()
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=scale_figsize(SINGLE_COL_WIDTH, 2.8))
    else:
        fig = ax.figure
    ax.plot(np.arange(window, len(ok) + 1), share)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Evaluation")
    ax.set_ylabel(f"Feasible share (last {window})")
    format_publication_axes(ax, x_integer=True)
    finalize_figure(fig, save_path, own_fig=own_fig, dpi=dpi)
    return ax


def plot_run_monitor(
    evaluated_objectives: np.ndarray,
    save_path: PathLike | None = None,
    *,
    eligible: np.ndarray | None = None,
    objective_names: Sequence[str] | None = None,
    maximize: Sequence[bool] | None = None,
    dpi: int | None = None,
):
    """One-file view of a running optimization: current front and hypervolume.

    The front is recomputed from the eligible evaluations so far; with more than
    two objectives the left panel shows the first two. Saved as PNG only, since
    it is overwritten throughout the run.

    :param evaluated_objectives: ``(n_evaluations, n_objectives)``, minimized.
    :param save_path: If provided, save ``<save_path>.png``.
    :param eligible: Mask of feasible, successful evaluations.
    :param objective_names: One label per objective.
    :param maximize: Per-objective display flags.
    :param dpi: Raster resolution.

    :returns: The Figure.
    """
    F = np.asarray(evaluated_objectives, dtype=float)
    ok = np.ones(len(F), bool) if eligible is None else np.asarray(eligible, bool)
    front = F[nondominated_eligible(F, ok)]
    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=scale_figsize(DOUBLE_COL_WIDTH, 2.8))
    names = None if objective_names is None else list(objective_names)[:2]
    plot_pareto_front_2d(front[:, :2], ax=left, objective_names=names,
                         maximize=None if maximize is None else list(maximize)[:2])
    plot_hypervolume_vs_evaluations(F, ax=right, eligible=ok, log_x=False)
    finalize_figure(fig, save_path, dpi=dpi, formats=("png",))
    return fig
