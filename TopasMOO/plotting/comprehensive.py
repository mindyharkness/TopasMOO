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

# Lean default set for finished runs. Optional keys (hypervolume) are skipped
# quietly when their source data is unavailable. Request ``"all"`` or name
# individual keys for parallel coordinates, decision heatmaps, etc.
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
    if len(pareto_objectives) == 0:
        logger.warning(
            "No Pareto solutions found. Skipping Pareto-dependent visualizations."
        )
    has_pareto = len(pareto_objectives) > 0

    n_obj = data.n_objectives

    # --- Pareto front --------------------------------------------------------
    if "pareto" in plots and has_pareto:
        plot_pareto_front(
            pareto_objectives,
            save_dir / "ParetoFront_Final",
            show_knee_point=True,
        )

    if "parallel" in plots and has_pareto:
        plot_parallel_coordinates(
            pareto_objectives,
            save_dir / "ParallelCoordinates_Final",
        )

    # --- Objective convergence -----------------------------------------------
    if "convergence" in plots:
        if data.log_file is not None:
            plot_objective_convergence(
                data.log_file,
                save_dir / "Convergence_Final",
                n_objectives=n_obj,
            )
        else:
            _warn_if_requested(
                explicit_request, "No log file available; skipping convergence plot."
            )

    # --- Parameter convergence -----------------------------------------------
    if "parameter_convergence" in plots:
        if data.log_file is not None:
            plot_parameter_convergence(
                data.log_file,
                save_dir / "ParameterConvergence_Final",
                parameter_names=data.parameter_names,
            )
        else:
            _warn_if_requested(
                explicit_request,
                "No log file available; skipping parameter convergence plot.",
            )

    # --- Hypervolume convergence ---------------------------------------------
    if "hypervolume" in plots:
        hv = data.hypervolume_history
        if hv:
            plot_hypervolume_convergence(
                hv,
                save_dir / "HypervolumeConvergence",
            )
        else:
            _warn_if_requested(
                explicit_request, "HypervolumeHistory not available; skipping hypervolume plot."
            )

    # --- Population evolution ------------------------------------------------
    if "population_evolution" in plots:
        pop_hist = data.population_history
        if pop_hist:
            plot_population_evolution(
                pop_hist,
                save_dir / "PopulationEvolution",
            )
        else:
            _warn_if_requested(
                explicit_request,
                "PopulationHistory not available; skipping population evolution plot.",
            )

    # --- Decision variable heatmap -------------------------------------------
    if "decision_heatmap" in plots:
        dec_vars = data.pareto_decision_vars
        if dec_vars is not None and len(dec_vars) > 0:
            plot_decision_heatmap(
                dec_vars,
                save_dir / "DecisionHeatmap",
                parameter_names=data.parameter_names,
            )
        else:
            _warn_if_requested(
                explicit_request, "ParetoDecisionVars not available; skipping decision heatmap."
            )

    # --- Petal diagrams ------------------------------------------------------
    if "petal" in plots:
        if has_pareto:
            plot_petal_diagram_multi(
                pareto_objectives,
                save_dir / "PetalDiagrams",
                title="Pareto Solutions Comparison",
            )
        else:
            _warn_if_requested(
                explicit_request,
                "No Pareto solutions available; skipping petal diagrams.",
            )

    # --- Parameter–objective correlation -------------------------------------
    if "correlation" in plots:
        dec_vars = data.pareto_decision_vars
        if dec_vars is not None and len(dec_vars) > 0:
            plot_parameter_objective_correlation(
                dec_vars,
                pareto_objectives,
                save_dir / "ParameterObjectiveCorrelation",
                parameter_names=data.parameter_names,
            )
        else:
            _warn_if_requested(
                explicit_request, "ParetoDecisionVars not available; skipping correlation plot."
            )

    # --- MOBO prospective GP prediction correlation -------------------------
    if "gp_correlation" in plots:
        observed = data.observed_objectives
        predicted = data.gp_prediction_history
        if observed is not None and predicted is not None:
            observed = np.asarray(observed, dtype=float)
            predicted = np.asarray(predicted, dtype=float)
            valid_mask = None
            if data.failed_mask is not None:
                failed = np.asarray(data.failed_mask, dtype=bool).reshape(-1)
                if len(failed) == len(observed):
                    valid_mask = ~failed
            has_predictions = observed.shape == predicted.shape and observed.ndim == 2
            if has_predictions:
                finite_pairs = np.isfinite(observed) & np.isfinite(predicted)
                if valid_mask is not None:
                    finite_pairs &= valid_mask[:, np.newaxis]
                has_predictions = bool(np.any(finite_pairs))
            if has_predictions:
                plot_gp_prediction_correlation(
                    observed,
                    predicted,
                    save_dir / "GPPredictionCorrelation",
                    valid_mask=valid_mask,
                )
            else:
                _warn_if_requested(
                    explicit_request,
                    "No prospective GP predictions available; skipping GP correlation plot.",
                )
        else:
            _warn_if_requested(
                explicit_request,
                "GP prediction history not available; skipping GP correlation plot.",
            )

    logger.info("Generated visualizations (%s) in %s", sorted(plots), save_dir)
