"""
Generate all visualizations for a completed optimization run.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence, Union

import numpy as np

from .convergence import plot_objective_convergence, plot_parameter_convergence
from .correlation import plot_parameter_objective_correlation
from .decision import plot_decision_heatmap
from .gp_correlation import plot_gp_prediction_correlation
from .hypervolume import plot_hypervolume_convergence
from .parallel import plot_parallel_coordinates
from .pareto import plot_pareto_front
from .petal import plot_petal_diagram_multi
from .population import plot_population_evolution

logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]


def _warn_if_requested(explicit_request: bool, message: str) -> None:
    """Warn about a skipped data-dependent plot only when explicitly requested.

    Optional plots (for example hypervolume) are skipped silently on default
    runs when their source data is unavailable; a warning is only useful when
    the user named the plot explicitly. Centralizing the rule here keeps every
    such skip consistent.
    """
    if explicit_request:
        logger.warning(message)

# Lean default set for finished runs.
# Request ``"all"`` or name individual keys for parallel coordinates, decision heatmaps, etc.
DEFAULT_FINAL_PLOTS = frozenset({
    "pareto",
    "convergence",
    "parameter_convergence",
    "hypervolume",
})

# All recognized plot keys.
ALL_PLOT_KEYS = frozenset({
    "pareto",
    "parallel",
    "convergence",
    "parameter_convergence",
    "hypervolume",
    "population_evolution",
    "decision_heatmap",
    "petal",
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
        return set(ALL_PLOT_KEYS), True
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
    can drive :func:`GenerateComprehensiveVisualizations` without mimicking
    ``TopasMOOBaseClass`` attributes.
    """

    pareto_objectives: np.ndarray
    n_objectives: int
    parameter_names: Sequence[str]
    log_file: PathLike | None = None
    pareto_decision_vars: np.ndarray | None = None
    hypervolume_history: Sequence[float] = field(default_factory=list)
    population_history: Sequence = field(default_factory=list)
    observed_objectives: np.ndarray | None = None
    gp_prediction_history: np.ndarray | None = None
    failed_mask: np.ndarray | None = None

    @classmethod
    def from_optimizer(cls, optimizer) -> "RunData":
        """Snapshot a completed ``TopasMOOBaseClass`` (or compatible) instance."""
        return cls(
            pareto_objectives=np.array(optimizer.ParetoObjectives),
            n_objectives=optimizer.n_objectives,
            parameter_names=optimizer.ParameterNames,
            log_file=getattr(optimizer, "_LogFileLoc", None),
            pareto_decision_vars=getattr(optimizer, "ParetoDecisionVars", None),
            hypervolume_history=getattr(optimizer, "HypervolumeHistory", []) or [],
            population_history=getattr(optimizer, "PopulationHistory", []) or [],
            observed_objectives=getattr(optimizer, "train_Y", None),
            gp_prediction_history=getattr(optimizer, "gp_prediction_history", None),
            failed_mask=getattr(optimizer, "train_failed", None),
        )


def _gp_correlation_inputs(data: RunData):
    """Return ``(observed, predicted, valid_mask)``, or ``(None, None, None)``
    when no finite prospective prediction pairs exist."""
    if data.observed_objectives is None or data.gp_prediction_history is None:
        return None, None, None
    observed = np.asarray(data.observed_objectives, dtype=float)
    predicted = np.asarray(data.gp_prediction_history, dtype=float)
    if observed.ndim != 2 or observed.shape != predicted.shape:
        return None, None, None
    valid_mask = None
    if data.failed_mask is not None:
        failed = np.asarray(data.failed_mask, dtype=bool).reshape(-1)
        if len(failed) == len(observed):
            valid_mask = ~failed
    finite_pairs = np.isfinite(observed) & np.isfinite(predicted)
    if valid_mask is not None:
        finite_pairs &= valid_mask[:, np.newaxis]
    if not finite_pairs.any():
        return None, None, None
    return observed, predicted, valid_mask


def GenerateComprehensiveVisualizations(
    run,
    save_dir: PathLike,
    final_plots: str | Iterable[str] | None = None,
) -> None:
    """Generate visualizations for a completed optimization run.

    :param run: A :class:`RunData` snapshot, or a ``TopasMOOBaseClass`` (or
        compatible) instance that has completed its ``RunOptimization``
        call (converted via :meth:`RunData.from_optimizer`).
    :param save_dir: Directory where plots will be saved.
    :param final_plots: Plot keys to generate. ``None`` or ``"default"``
        expands to :data:`DEFAULT_FINAL_PLOTS` (Pareto, objective/parameter
        convergence, and hypervolume when history is available). A MOBO run
        with prospective prediction history also adds ``"gp_correlation"``.
        Pass
        ``"all"`` for every key, a single recognized key string such as
        ``"pareto"``, or an iterable of keys. Optional plots are skipped
        when their data is unavailable. Recognized keys:

        * ``"pareto"`` — 2D / 3D / pairwise Pareto front (auto-selected)
        * ``"parallel"`` — parallel coordinates for multi-objective trade-offs
        * ``"convergence"`` — objective convergence line plot
        * ``"parameter_convergence"`` — parameter convergence line plot
        * ``"hypervolume"`` — hypervolume vs. generation
        * ``"population_evolution"`` — population snapshots over generations
        * ``"decision_heatmap"`` — normalized parameter heatmap + boxplots
        * ``"petal"`` — Nightingale petal diagrams (≥3 objectives)
        * ``"correlation"`` — parameter–objective scatter grid
        * ``"gp_correlation"`` — prospective GP predictions vs. observations
    """
    data = run if isinstance(run, RunData) else RunData.from_optimizer(run)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plots, explicit_request = _resolve_final_plots(final_plots)
    # Prospective GP correlation is a high-value default for MOBO but has no
    # meaning for NSGA-II. Add it only when an optimizer supplied prediction
    # history, leaving the shared default set and NSGA-II output unchanged.
    if not explicit_request and data.gp_prediction_history is not None:
        predicted = np.asarray(data.gp_prediction_history, dtype=float)
        if predicted.size and np.any(np.isfinite(predicted)):
            plots.add("gp_correlation")

    pareto_objectives = np.asarray(data.pareto_objectives)
    has_pareto = len(pareto_objectives) > 0
    if not has_pareto:
        logger.warning(
            "No Pareto solutions found. Skipping Pareto-dependent visualizations."
        )
    dec_vars = data.pareto_decision_vars
    has_dec_vars = dec_vars is not None and len(dec_vars) > 0
    gp_observed, gp_predicted, gp_valid = _gp_correlation_inputs(data)

    # key -> (data available, draw, message when an explicit request is skipped).
    # Pareto and parallel plots are covered by the "No Pareto solutions" warning.
    specs = {
        "pareto": (has_pareto, lambda: plot_pareto_front(
            pareto_objectives, save_dir / "ParetoFront_Final", show_knee_point=True
        ), None),
        "parallel": (has_pareto, lambda: plot_parallel_coordinates(
            pareto_objectives, save_dir / "ParallelCoordinates_Final"
        ), None),
        "convergence": (data.log_file is not None, lambda: plot_objective_convergence(
            data.log_file, save_dir / "Convergence_Final", n_objectives=data.n_objectives
        ), "No log file available; skipping convergence plot."),
        "parameter_convergence": (data.log_file is not None, lambda: plot_parameter_convergence(
            data.log_file,
            save_dir / "ParameterConvergence_Final",
            parameter_names=data.parameter_names,
        ), "No log file available; skipping parameter convergence plot."),
        "hypervolume": (bool(data.hypervolume_history), lambda: plot_hypervolume_convergence(
            data.hypervolume_history, save_dir / "HypervolumeConvergence"
        ), "HypervolumeHistory not available; skipping hypervolume plot."),
        "population_evolution": (bool(data.population_history), lambda: plot_population_evolution(
            data.population_history, save_dir / "PopulationEvolution"
        ), "PopulationHistory not available; skipping population evolution plot."),
        "decision_heatmap": (has_dec_vars, lambda: plot_decision_heatmap(
            dec_vars, save_dir / "DecisionHeatmap", parameter_names=data.parameter_names
        ), "ParetoDecisionVars not available; skipping decision heatmap."),
        "petal": (has_pareto, lambda: plot_petal_diagram_multi(
            pareto_objectives, save_dir / "PetalDiagrams", title="Pareto Solutions Comparison"
        ), "No Pareto solutions available; skipping petal diagrams."),
        "correlation": (has_dec_vars, lambda: plot_parameter_objective_correlation(
            dec_vars,
            pareto_objectives,
            save_dir / "ParameterObjectiveCorrelation",
            parameter_names=data.parameter_names,
        ), "ParetoDecisionVars not available; skipping correlation plot."),
        "gp_correlation": (gp_observed is not None, lambda: plot_gp_prediction_correlation(
            gp_observed, gp_predicted, save_dir / "GPPredictionCorrelation", valid_mask=gp_valid
        ), "No prospective GP predictions available; skipping GP correlation plot."),
    }
    for key, (available, draw, skip_message) in specs.items():
        if key not in plots:
            continue
        if available:
            draw()
        elif skip_message:
            _warn_if_requested(explicit_request, skip_message)

    logger.info("Generated visualizations (%s) in %s", sorted(plots), save_dir)
