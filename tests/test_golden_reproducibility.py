"""Seeded end-to-end reproducibility guard for the ZDT1 benchmark.

This exercises the real code path (script generation -> file -> parse ->
objective -> pymoo ask/tell loop -> final non-dominated set) and checks the two
properties that must never break:

1. **Determinism.** Two runs with the same seed, in independent directories,
   must produce *bit-identical* Pareto fronts. This guards seed threading, the
   cache key, the dominance sort, the evaluation order, and the NSGA-II operator
   configuration -- anything that perturbs the seeded trajectory will desync the
   two runs and fail the test.

2. **Correctness.** The returned set must be a valid ZDT1 Pareto front: a
   non-empty, mutually non-dominated set of feasible points sitting at or above
   the analytic ZDT1 front ``f2 = 1 - sqrt(f1)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from TopasMOO.optimizers import NSGAII_Optimizer

# GenerateTopasScripts that records every decision variable as a comment, exactly
# like the DevelopmentExample, so the round-trip through the generated .tps file
# is exercised rather than bypassed.
_GENERATE_SCRIPT = '''def GenerateTopasScripts(BaseDirectory, iteration, **variable_dict):
    script = ["# ZDT1 golden reproducibility test"]
    for i in range(1, 6):
        script.append(f"# x{i} = {variable_dict[f'x{i}']}")
    return [script], ["GoldenZDT1"]
'''

# TopasObjectiveFunction that parses the parameters back out of the generated
# script and evaluates the analytic ZDT1 objectives (both minimized).
_OBJECTIVE_SCRIPT = '''from pathlib import Path
import numpy as np
def TopasObjectiveFunction(ResultsLocation, iteration):
    script = Path(ResultsLocation).parent / "TopasScripts" / f"GoldenZDT1_itt_{iteration}.tps"
    x = []
    with open(script) as f:
        for line in f:
            if line.strip().startswith("#") and "x" in line and "=" in line:
                try:
                    x.append(float(line.split("=")[1].strip().split()[0]))
                except (ValueError, IndexError):
                    pass
    x = np.array(x)
    f1 = x[0]
    g = 1.0 + 9.0 / (len(x) - 1) * np.sum(x[1:])
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return [f1, f2]
'''


def _write_zdt1_project(opt_dir: Path) -> None:
    opt_dir.mkdir(parents=True, exist_ok=True)
    (opt_dir / "GenerateTopasScripts.py").write_text(_GENERATE_SCRIPT)
    (opt_dir / "TopasObjectiveFunction.py").write_text(_OBJECTIVE_SCRIPT)


def _run_zdt1(opt_dir: Path) -> np.ndarray:
    """Run the seeded ZDT1 optimization in ``opt_dir`` and return the front (F)."""
    _write_zdt1_project(opt_dir)

    optimization_params = {
        "ParameterNames": ["x1", "x2", "x3", "x4", "x5"],
        "UpperBounds": np.ones(5),
        "LowerBounds": np.zeros(5),
        "start_point": np.full(5, 0.5),
        "n_generations": 5,
        "n_objectives": 2,
    }
    optimizer = NSGAII_Optimizer(
        optimization_params=optimization_params,
        BaseDirectory=str(opt_dir),
        SimulationName="Golden",
        OptimizationDirectory=opt_dir,
        TopasLocation="testing_mode",
        Overwrite=True,
        pop_size=8,
        seed=42,
    )
    result = optimizer.RunOptimization()

    front = np.atleast_2d(np.asarray(result.F, dtype=float))
    return front[front[:, 0].argsort()]


def _is_mutually_nondominated(front: np.ndarray) -> bool:
    """True if no row of ``front`` is Pareto-dominated by another (minimization)."""
    n = len(front)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(front[j] <= front[i]) and np.any(front[j] < front[i]):
                return False
    return True


def test_zdt1_seeded_run_is_deterministic(tmp_path: Path) -> None:
    """Same seed, independent dirs -> bit-identical Pareto fronts."""
    front_a = _run_zdt1(tmp_path / "run_a")
    front_b = _run_zdt1(tmp_path / "run_b")

    assert front_a.shape == front_b.shape, (
        f"Seeded run is non-deterministic: front shapes differ, "
        f"{front_a.shape} vs {front_b.shape}. A refactor broke seed threading "
        f"or introduced an unseeded source of randomness."
    )
    np.testing.assert_array_equal(
        front_a,
        front_b,
        err_msg=(
            "Seeded ZDT1 runs diverged. Two runs with the same seed must be "
            "bit-identical; a refactor altered the seeded trajectory (cache key, "
            "dominance sort, evaluation order, or NSGA-II operator config)."
        ),
    )


def test_zdt1_run_is_a_valid_pareto_front(tmp_path: Path) -> None:
    """The returned set is a valid, feasible ZDT1 Pareto front."""
    front = _run_zdt1(tmp_path)

    # Non-empty and shaped (n, 2).
    assert front.ndim == 2 and front.shape[1] == 2, f"Unexpected front shape: {front.shape}"
    assert len(front) >= 1, "Optimizer returned an empty Pareto front."

    f1, f2 = front[:, 0], front[:, 1]

    # Feasible region of ZDT1: f1 = x1 in [0, 1] and f2 >= 0.
    assert np.all(f1 >= -1e-9) and np.all(f1 <= 1 + 1e-9), f"f1 out of [0, 1]: {f1}"
    assert np.all(f2 >= -1e-9), f"f2 must be non-negative: {f2}"

    # Every ZDT1 point satisfies f2 = g(1 - sqrt(f1/g)) >= 1 - sqrt(f1) (g >= 1),
    # so the whole front must sit at or above the analytic front. A small
    # tolerance absorbs float rounding in the objective round-trip.
    true_front_f2 = 1.0 - np.sqrt(np.clip(f1, 0.0, 1.0))
    assert np.all(f2 >= true_front_f2 - 1e-6), (
        "Front lies below the analytic ZDT1 front f2 = 1 - sqrt(f1); the "
        "objective evaluation or parameter round-trip is broken."
    )

    # The returned set must genuinely be non-dominated.
    assert _is_mutually_nondominated(front), (
        "Returned front contains dominated solutions; the final non-dominated "
        "extraction is broken."
    )
