"""
Multi-objective optimization metrics for Pareto front analysis.

All metrics assume minimization of objectives.
"""

from __future__ import annotations

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def normalize_objectives(
    objectives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max normalize objectives per column to ``[0, 1]``.

    Each objective (column) is scaled by its observed range across the given
    solutions. Columns with zero range (a single distinct value) are left at
    ``0`` rather than dividing by zero.

    :param objectives: Array of shape ``(n_solutions, n_objectives)``.

    :returns: Tuple ``(normalized, ideal, nadir)`` where ``ideal`` and ``nadir`` are
        the per-objective minimum and maximum, and ``normalized`` has the same
        shape as ``objectives``.
    """
    objectives = np.asarray(objectives, dtype=float)
    ideal = objectives.min(axis=0)
    nadir = objectives.max(axis=0)
    obj_range = nadir - ideal
    obj_range[obj_range == 0] = 1
    normalized = (objectives - ideal) / obj_range
    return normalized, ideal, nadir


def hypervolume_reference_point(objectives: np.ndarray) -> np.ndarray:
    """Return a hypervolume reference dominated by every given solution.

    The point is the observed nadir pushed out by ten percent of the observed
    span, so every solution contributes positive volume. Objectives with zero
    span (a single distinct value) use a unit margin rather than collapsing the
    box to zero width.

    A single shared definition matters because the margin sets the hypervolume
    scale: histories computed with different margins are not comparable.

    :param objectives: Array of shape ``(n_solutions, n_objectives)``, minimized.

    :returns: Reference point of shape ``(n_objectives,)``.
    """
    objectives = np.asarray(objectives, dtype=float)
    if objectives.ndim != 2 or objectives.shape[0] == 0 or objectives.shape[1] == 0:
        raise ValueError(
            "objectives must have shape (n_solutions, n_objectives) with n_solutions > 0."
        )
    ideal = objectives.min(axis=0)
    nadir = objectives.max(axis=0)
    span = nadir - ideal
    span[span == 0] = 1.0
    return nadir + 0.1 * span


def calculate_knee_point(pareto_objectives: np.ndarray) -> int:
    """Find the knee point (best trade-off) on a Pareto front.

    Uses a trade-off method: the knee is the solution with the minimum
    sum of min-max normalized objectives, representing the best balanced
    compromise across all objectives.

    :param pareto_objectives: Array of shape ``(n_solutions, n_objectives)``.
        All objectives are assumed to be minimized.

    :returns: Index (int) of the knee point solution in the input array.
    """
    normalized, _ideal, _nadir = normalize_objectives(pareto_objectives)

    total_objectives = normalized.sum(axis=1)
    return int(np.argmin(total_objectives))


def calculate_crowding_distance(pareto_objectives: np.ndarray) -> np.ndarray:
    """Calculate NSGA-II crowding distance for each solution.

    Crowding distance measures how isolated a solution is in objective
    space.  Higher values indicate solutions in less crowded regions,
    which are preferred during selection to maintain diversity.

    Boundary solutions (best/worst in any single objective) receive
    infinite crowding distance.

    :param pareto_objectives: Array of shape ``(n_solutions, n_objectives)``.

    :returns: Array of crowding distances with shape ``(n_solutions,)``.
    """
    n_solutions, n_objectives = pareto_objectives.shape

    if n_solutions <= 2:
        return np.full(n_solutions, np.inf)

    crowding = np.zeros(n_solutions)

    for m in range(n_objectives):
        sorted_indices = np.argsort(pareto_objectives[:, m])
        obj_values = pareto_objectives[sorted_indices, m]

        crowding[sorted_indices[0]] = np.inf
        crowding[sorted_indices[-1]] = np.inf

        obj_range = obj_values[-1] - obj_values[0]
        if obj_range == 0:
            continue

        for i in range(1, n_solutions - 1):
            crowding[sorted_indices[i]] += (
                obj_values[i + 1] - obj_values[i - 1]
            ) / obj_range

    return crowding


def calculate_dominance_rank(objectives: np.ndarray) -> np.ndarray:
    """Assign dominance ranks using fast non-dominated sorting.

    Rank 0 contains the Pareto front (non-dominated solutions).
    Rank 1 contains solutions dominated only by rank-0 solutions,
    and so on.

    All objectives are assumed to be minimized.

    :param objectives: Array of shape ``(n_solutions, n_objectives)``.

    :returns: Integer array of dominance ranks with shape ``(n_solutions,)``.
    """
    return NonDominatedSorting().do(np.asarray(objectives, dtype=float), return_rank=True)[1]
