"""ZDT1 objectives for the MOBO development example (testing_mode).

Reference:
    Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of multiobjective
    evolutionary algorithms: Empirical results.
"""

import os
from pathlib import Path

import numpy as np

#: Must match ``PARAM_PREFIX`` in ``GenerateTopasScripts.py``. Deliberately a
#: local literal rather than an import: the two user scripts are loaded from an
#: absolute path under unique module names, and importing a sibling by its bare
#: name would collide across optimization directories in the same process.
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
    script_dir = Path(ResultsLocation).parent / "TopasScripts"
    script_file = script_dir / f"MOBODevelopmentExample_itt_{iteration}.tps"
    x = []
    if os.path.exists(script_file):
        with open(script_file) as f:
            for line in f:
                # Only PARAM-tagged lines are decision variables; any other
                # comment containing "=" is ignored rather than parsed as one.
                if not line.startswith(PARAM_PREFIX):
                    continue
                try:
                    x.append(float(line.split("=")[1].strip().split()[0]))
                except (ValueError, IndexError):
                    pass
    if len(x) < 2:
        # ZDT1 needs at least two variables for g to be defined.
        x = [0.5] * 5
    return ZDT1(np.array(x))
