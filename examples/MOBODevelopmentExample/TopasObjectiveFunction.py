"""ZDT1 objectives for the MOBO development example (testing_mode).

Reference:
    Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of multiobjective
    evolutionary algorithms: Empirical results.
"""

from pathlib import Path

import numpy as np

PARAM_PREFIX = "# PARAM "


def ZDT1(x):
    """Evaluate the ZDT1 benchmark objectives for a decision vector.

    ``f1 = x1``, ``g = 1 + 9 * sum(x2..xn) / (n - 1)``, ``h = 1 - sqrt(f1 / g)``,
    ``f2 = g * h``. Known Pareto front: ``f2 = 1 - sqrt(f1)`` for ``f1`` in
    ``[0, 1]``. Both objectives are minimized.

    :param x: Decision variables (array of length n).

    :returns: List ``[f1, f2]`` of the two objective values.
    """
    x = np.atleast_1d(x)
    n = len(x)
    f1 = x[0]
    g = 1.0 + (9.0 / (n - 1)) * np.sum(x[1:]) if n > 1 else 1.0
    h = 1.0 - np.sqrt(f1 / g)
    return [f1, g * h]


def TopasObjectiveFunction(ResultsLocation, iteration):
    """Compute the ZDT1 objectives for one optimization iteration.

    For this example no real TOPAS output is read: the decision variables are
    recovered from the comments ``GenerateTopasScripts`` wrote, in the same
    order they were written (the optimizer's ``ParameterNames`` order), and the
    analytic objectives are evaluated. A real study would instead read results
    from ``ResultsLocation`` and derive the competing objectives.

    :param ResultsLocation: Path to the iteration's results directory.
    :param iteration: Current iteration number.

    :returns: List ``[f1, f2]`` of the two ZDT1 objective values.
    """
    script_file = Path(ResultsLocation).parent / "TopasScripts" / f"MOBODevelopmentExample_itt_{iteration}.tps"
    with open(script_file) as f:
        # Only PARAM-tagged lines are decision variables; any other comment
        # containing "=" is ignored rather than parsed as one.
        x = [float(line.split("=")[1].split()[0]) for line in f if line.startswith(PARAM_PREFIX)]
    if len(x) < 2:
        raise ValueError(f"{script_file} has {len(x)} PARAM lines; ZDT1 needs at least 2.")
    return ZDT1(np.array(x))
