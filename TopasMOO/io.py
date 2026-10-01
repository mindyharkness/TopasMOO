"""
I/O utilities for reading and writing TopasMOO optimization logs.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from typing import Union

import numpy as np

from .exceptions import MalformedOutputError

PathLike = Union[str, os.PathLike]


def ReadInMultiObjectiveLogFile(LogFilePath: PathLike) -> dict[str, list[float]]:
    """Read a multi-objective optimization log file into a dictionary.

    Parses the comma-separated log format produced by TopasMOO during
    optimization, extracting iteration numbers, parameter values, and
    objective function values.

    :param LogFilePath: Path to the optimization log file
        (typically ``logs/OptimizationLogs.txt``).

    :returns: Dictionary mapping column names to lists of float values.
        Keys include ``'Iteration'``, parameter names, and
        ``'ObjectiveFunction_1'``, ``'ObjectiveFunction_2'``, etc.

    :raises FileNotFoundError: If the log file does not exist (from ``open``).
    :raises MalformedOutputError: If a data row contains a non-numeric value
        after the ``key: value`` split.
    """
    log_path = os.fspath(LogFilePath)
    results: dict[str, list[float]] = {}
    with open(log_path, "r") as f:
        lines = f.readlines()

    for line_number, line in enumerate(lines, start=1):
        if not line.startswith("Iteration"):
            continue
        for entry in line.split(","):
            parts = entry.split(":")
            if len(parts) < 2:
                continue

            key = parts[0].strip()
            raw_value = parts[1].strip()
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise MalformedOutputError(
                    f"Could not parse value '{raw_value}' for key '{key}' "
                    f"on line {line_number} of {log_path}"
                ) from exc

            results.setdefault(key, []).append(value)

    return results


def LogParetoFrontToFile(
    LogFilePath: PathLike,
    ParetoObjectives: np.ndarray | Iterable[Iterable[float]],
    ParameterNames: Sequence[str],
    n_objectives: int,
    ParetoDecisionVars: np.ndarray | Iterable[Iterable[float]] | None = None,
) -> None:
    """Write the current Pareto front to a CSV-style log file.

    :param LogFilePath: Output file path.
    :param ParetoObjectives: Array of shape ``(n_solutions, n_objectives)``
        containing objective values for all non-dominated solutions.
    :param ParameterNames: Decision-variable names, used as the column headers for
        the decision-variable values when ``ParetoDecisionVars`` is given.
    :param n_objectives: Number of objective functions.
    :param ParetoDecisionVars: Optional array of shape
        ``(n_solutions, len(ParameterNames))`` aligned row-for-row with
        ``ParetoObjectives``. When provided, each parameter is written as an
        additional column so the file fully describes each solution. When
        ``None``, only objective columns are written.
    """
    objectives = np.asarray(ParetoObjectives, dtype=float).reshape(-1, n_objectives)
    columns = [np.arange(len(objectives)), objectives]
    header = ["Solution_Index"] + [f"Objective_{i + 1}" for i in range(n_objectives)]
    if ParetoDecisionVars is not None:
        columns.append(
            np.asarray(ParetoDecisionVars, dtype=float).reshape(
                len(objectives), len(ParameterNames)
            )
        )
        header += list(ParameterNames)
    table = np.column_stack(columns)
    fmt = ["%d"] + ["%.6f"] * (table.shape[1] - 1)
    np.savetxt(os.fspath(LogFilePath), table, fmt=fmt, delimiter=",",
               header=",".join(header), comments="")
