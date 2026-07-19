"""
I/O utilities for reading and writing TopasMOO optimization logs.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Union

from .exceptions import MalformedOutputError

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

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

    :raises FileNotFoundError: If the log file does not exist.
    :raises MalformedOutputError: If a data row contains a non-numeric value
        after the ``key: value`` split.
    """
    log_path = os.fspath(LogFilePath)
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Could not find log file at {log_path}")

    results: dict[str, list[float]] = {}
    with open(log_path, "r") as f:
        lines = f.readlines()

    if not lines:
        return results

    if not any("Iteration" in line for line in lines):
        logger.warning("Log file format not recognized or empty: %s", log_path)
        return results

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
    with open(os.fspath(LogFilePath), "w") as f:
        header = "Solution_Index"
        for i in range(n_objectives):
            header += f",Objective_{i+1}"
        if ParetoDecisionVars is not None:
            for name in ParameterNames:
                header += f",{name}"
        header += "\n"
        f.write(header)

        decision_rows = list(ParetoDecisionVars) if ParetoDecisionVars is not None else None
        for i, objectives in enumerate(ParetoObjectives):
            line = f"{i}"
            for obj_val in objectives:
                line += f",{obj_val:.6f}"
            if decision_rows is not None:
                for var_val in decision_rows[i]:
                    line += f",{var_val:.6f}"
            line += "\n"
            f.write(line)
