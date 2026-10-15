"""
Generate the end-of-run figures for a completed optimization.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Union

import numpy as np

from .convergence import plot_parameter_convergence
from .correlation import plot_parameter_objective_correlation
from .decision import plot_decision_along_front
from .gp_correlation import plot_gp_prediction_correlation
from .parallel import plot_parallel_coordinates
from .pareto import plot_pareto_front, plot_pareto_front_2d
from .progress import plot_constraint_check, plot_hypervolume_vs_evaluations
from .table import plot_solution_table

logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]

# Written at the end of every run: the Pareto front (pair projections for 3+
# objectives). Everything else is a diagnostic and opt-in.
DEFAULT_FINAL_PLOTS = frozenset({"pareto"})

# All recognized plot keys.
ALL_PLOT_KEYS = frozenset({
    "pareto",
    "pareto_context",
    "hypervolume",
    "constraints",
    "solution_table",
    "parallel",
    "decision",
    "parameter_convergence",
    "correlation",
    "gp_correlation",
})


def _resolve_final_plots(
    final_plots: str | Iterable[str] | None,
) -> tuple[set[str], bool]:
    """Expand ``final_plots`` into a key set and whether the request was explicit.

    A bare string is treated as a single key (or the aliases ``"default"`` /
    ``"all"``), never as an iterable of characters. That avoids the footgun
    where ``set("pareto")`` silently becomes ``{'p', 'a', 'r', ...}`` and
    produces no figures.
    """
    if final_plots is None or final_plots == "default":
        return set(DEFAULT_FINAL_PLOTS), False
    if final_plots == "all":
        # Not "explicit": a run simply lacks data for some figures (e.g. no
        # constraints), which is not worth a warning.
        return set(ALL_PLOT_KEYS), False
    if isinstance(final_plots, str):
        if final_plots in ALL_PLOT_KEYS:
            return {final_plots}, True
        logger.warning("Unrecognized final_plots key (ignored): %s", final_plots)
        return set(), True

    requested = set(final_plots)
    unknown = requested - ALL_PLOT_KEYS
    if unknown:
        logger.warning("Unrecognized final_plots keys (ignored): %s", unknown)
    return requested & ALL_PLOT_KEYS, True


@dataclass
class RunData:
    """Plain-data snapshot of a finished optimization run, for plotting.

    Decouples the figure orchestration from the optimizer object: anything able
    to supply these arrays (a live optimizer, parsed log files, a test harness)
    can drive :func:`GenerateComprehensiveVisualizations`. All objectives are
    in minimized form; ``maximize`` only changes how they are displayed.
    """

    pareto_objectives: np.ndarray
    n_objectives: int
    parameter_names: Sequence[str]
    pareto_decision_vars: np.ndarray | None = None
    evaluated_objectives: np.ndarray | None = None
    eligible: np.ndarray | None = None
    failed: np.ndarray | None = None
    n_constraints: int = 0
    lower_bounds: Sequence[float] | None = None
    upper_bounds: Sequence[float] | None = None
    objective_names: Sequence[str] | None = None
    maximize: Sequence[bool] | None = None
    log_file: PathLike | None = None
    gp_prediction_history: np.ndarray | None = None

    @classmethod
    def from_optimizer(cls, optimizer) -> "RunData":
        """Snapshot a ``TopasMOOBaseClass`` (or compatible) instance."""
        history = getattr(optimizer, "evaluation_history", None)
        evaluated, eligible, failed = history() if history else (None, None, None)
        return cls(
            pareto_objectives=np.array(optimizer.ParetoObjectives),
            n_objectives=optimizer.n_objectives,
            parameter_names=optimizer.ParameterNames,
            pareto_decision_vars=getattr(optimizer, "ParetoDecisionVars", None),
            evaluated_objectives=evaluated,
            eligible=eligible,
            failed=failed,
            n_constraints=getattr(optimizer, "n_constraints", 0),
            lower_bounds=getattr(optimizer, "LowerBounds", None),
            upper_bounds=getattr(optimizer, "UpperBounds", None),
            objective_names=getattr(optimizer, "ObjectiveNames", None),
            maximize=getattr(optimizer, "Maximize", None),
            log_file=getattr(optimizer, "_LogFileLoc", None),
            gp_prediction_history=getattr(optimizer, "gp_prediction_history", None),
        )


def GenerateComprehensiveVisualizations(
    run,
    save_dir: PathLike,
    final_plots: str | Iterable[str] | None = None,
) -> None:
    """Write the end-of-run figures for a completed optimization.

    :param run: A :class:`RunData` snapshot, or a ``TopasMOOBaseClass`` (or
        compatible) instance that has completed its ``RunOptimization`` call.
    :param save_dir: Directory where figures are saved (PDF + PNG each).
    :param final_plots: ``None`` / ``"default"`` writes only the Pareto front
        (:data:`DEFAULT_FINAL_PLOTS`); ``"all"`` writes every figure the run has
        data for; a key or an iterable of keys selects figures:

        * ``"pareto"`` — Pareto front and knee point (pair projections for 3+
          objectives)
        * ``"pareto_context"`` — 2-objective front over all other designs (gray)
        * ``"hypervolume"`` — hypervolume vs evaluations
        * ``"constraints"`` — share of feasible designs (constrained runs only)
        * ``"solution_table"`` — knee point and best design per objective
        * ``"parallel"`` — parameters and objectives in real units
        * ``"decision"`` — parameter positions within bounds along the front
        * ``"parameter_convergence"`` — parameter values over evaluations
        * ``"correlation"`` — parameter vs objective scatter grid
        * ``"gp_correlation"`` — MOBO GP predictions vs observations

        Figures without data are skipped; a warning is logged only for keys
        that were named explicitly.
    """
    data = run if isinstance(run, RunData) else RunData.from_optimizer(run)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plots, explicit = _resolve_final_plots(final_plots)

    front = np.asarray(data.pareto_objectives, dtype=float).reshape(-1, data.n_objectives)
    has_front = len(front) > 0
    if not has_front:
        logger.warning("No Pareto solutions found. Skipping Pareto-dependent visualizations.")
    X = data.pareto_decision_vars
    has_x = X is not None and len(X) > 0
    has_bounds = data.lower_bounds is not None and data.upper_bounds is not None
    has_history = data.evaluated_objectives is not None and len(data.evaluated_objectives) > 0
    names = dict(objective_names=data.objective_names, maximize=data.maximize)
    gp_ok = (
        data.gp_prediction_history is not None
        and has_history
        and np.shape(data.gp_prediction_history) == np.shape(data.evaluated_objectives)
        and np.isfinite(data.gp_prediction_history).any()
    )

    # key -> (data available, draw, message when an explicit request is skipped)
    specs = {
        "pareto": (has_front, lambda: plot_pareto_front(
            front, save_dir / "ParetoFront_Final", **names), None),
        "pareto_context": (has_front and has_history and data.n_objectives == 2,
                           lambda: plot_pareto_front_2d(
            front, save_dir / "ParetoFront_Context", evaluated_objectives=data.evaluated_objectives,
            eligible=data.eligible, **names),
            "The context plot needs 2 objectives and the evaluation history; skipping."),
        "hypervolume": (has_history, lambda: plot_hypervolume_vs_evaluations(
            data.evaluated_objectives, save_dir / "HypervolumeVsEvaluations", eligible=data.eligible),
            "No evaluation history available; skipping hypervolume plot."),
        "constraints": (has_history and data.n_constraints > 0 and data.eligible is not None,
                        lambda: plot_constraint_check(data.eligible, save_dir / "ConstraintCheck"),
            "No constraints (n_constraints = 0); skipping constraint check."),
        "solution_table": (has_front and has_x, lambda: plot_solution_table(
            front, X, save_dir / "SolutionTable", parameter_names=data.parameter_names, **names),
            "ParetoDecisionVars not available; skipping solution table."),
        "parallel": (has_front, lambda: plot_parallel_coordinates(
            front, save_dir / "ParallelCoordinates_Final", decision_vars=X if has_x else None,
            lower_bounds=data.lower_bounds, upper_bounds=data.upper_bounds,
            parameter_names=data.parameter_names, **names), None),
        "decision": (has_front and has_x and has_bounds, lambda: plot_decision_along_front(
            front, X, save_dir / "DecisionVariables", lower_bounds=data.lower_bounds,
            upper_bounds=data.upper_bounds, parameter_names=data.parameter_names,
            objective_names=data.objective_names),
            "Decision variables or parameter bounds not available; skipping decision plot."),
        "parameter_convergence": (data.log_file is not None, lambda: plot_parameter_convergence(
            data.log_file, save_dir / "ParameterConvergence_Final",
            parameter_names=data.parameter_names),
            "No log file available; skipping parameter convergence plot."),
        "correlation": (has_front and has_x, lambda: plot_parameter_objective_correlation(
            X, front, save_dir / "ParameterObjectiveCorrelation",
            parameter_names=data.parameter_names, **names),
            "ParetoDecisionVars not available; skipping correlation plot."),
        "gp_correlation": (gp_ok, lambda: plot_gp_prediction_correlation(
            data.evaluated_objectives, data.gp_prediction_history,
            save_dir / "GPPredictionCorrelation",
            valid_mask=None if data.failed is None else ~np.asarray(data.failed, bool), **names),
            "No prospective GP predictions available; skipping GP correlation plot."),
    }
    for key, (available, draw, skip_message) in specs.items():
        if key not in plots:
            continue
        if available:
            draw()
        elif skip_message and explicit:
            logger.warning(skip_message)

    logger.info("Generated visualizations (%s) in %s", sorted(plots), save_dir)
