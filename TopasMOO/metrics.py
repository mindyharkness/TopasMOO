"""
Multi-objective optimization metrics for Pareto front analysis.

All metrics assume minimization of objectives.
"""

from __future__ import annotations

import numpy as np


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
    n_solutions = len(objectives)
    domination_count = np.zeros(n_solutions, dtype=int)
    dominated_solutions = [[] for _ in range(n_solutions)]
    ranks = np.zeros(n_solutions, dtype=int)

    for i in range(n_solutions):
        for j in range(i + 1, n_solutions):
            if np.all(objectives[i] <= objectives[j]) and np.any(
                objectives[i] < objectives[j]
            ):
                dominated_solutions[i].append(j)
                domination_count[j] += 1
            elif np.all(objectives[j] <= objectives[i]) and np.any(
                objectives[j] < objectives[i]
            ):
                dominated_solutions[j].append(i)
                domination_count[i] += 1

    current_front = np.where(domination_count == 0)[0]
    rank = 0

    while len(current_front) > 0:
        ranks[current_front] = rank
        next_front = []

        for i in current_front:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)

        current_front = np.array(next_front)
        rank += 1

    return ranks
