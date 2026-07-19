"""
Evaluate the ZDT1 benchmark objectives for a decision vector. 
Reference:
        Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of multiobjective
        evolutionary algorithms: Empirical results.
"""

import os
from pathlib import Path

import numpy as np


def ZDT1(x):
    """Evaluate the ZDT1 benchmark objectives for a decision vector.

    ZDT1 is a standard multi-objective test function with a known Pareto front,
    making it ideal for validating that the optimizer works correctly. Both
    objectives are minimized.

    Definition:
        ``f1(x) = x1``
        ``g(x)  = 1 + 9 * sum(x2..xn) / (n - 1)``
        ``h     = 1 - sqrt(f1 / g)``
        ``f2    = g * h``
    
    Known Pareto front: ``f2 = 1 - sqrt(f1)`` for ``f1`` in ``[0, 1]``.

    :param x: Decision variables (array of length n).

    :returns: List ``[f1, f2]`` of the two objective values.
    """
    x = np.atleast_1d(x)
    n = len(x)

    # Objective 1: simply the first variable.
    f1 = x[0]

    # Helper function g.
    if n > 1:
        g = 1.0 + (9.0 / (n - 1)) * np.sum(x[1:])
    else:
        g = 1.0

    # Helper function h, then objective 2.
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h

    return [f1, f2]


def TopasObjectiveFunction(ResultsLocation, iteration):
    """Compute the ZDT1 objectives for one optimization iteration.

    This is the objective entry point the framework calls. For the development
    example it does not read real TOPAS output; instead it recovers the decision
    variables from the comments written by ``GenerateTopasScripts`` and evaluates
    the analytic ZDT1 objectives, so the full optimization loop can be exercised
    without running expensive simulations.

    A real study would instead read TOPAS results from ``ResultsLocation`` (for
    example with ``topas2numpy`` / ``TopasOpt`` helpers), derive the competing
    objectives, and return them as a list.

    :param ResultsLocation: Path to the iteration's results directory.
    :param iteration: Current iteration number.

    :returns: List ``[f1, f2]`` of the two ZDT1 objective values (both minimized).
    """
    # Recover the parameter values from the generated TOPAS script comments.
    script_dir = Path(ResultsLocation).parent / "TopasScripts"
    script_file = script_dir / f"DevelopmentExample_itt_{iteration}.tps"

    x = []
    if os.path.exists(script_file):
        with open(script_file, "r") as f:
            for line in f:
                if "x" in line.lower() and "=" in line and line.strip().startswith("#"):
                    try:
                        value = float(line.split("=")[1].strip().split()[0])
                        x.append(value)
                    except (ValueError, IndexError):
                        pass

    # Fallback to mid-range values if parsing failed (should not happen).
    if len(x) == 0:
        x = [0.5, 0.5, 0.5, 0.5, 0.5]

    return ZDT1(np.array(x))
