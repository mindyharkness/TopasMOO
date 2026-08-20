"""Shared helpers for MOBO benchmark scripts (not imported by the library)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from TopasMOO.metrics import hypervolume_reference_point
from TopasMOO.mobo import MOBOOptimizer

__all__ = [
    "hypervolume_reference_point",
    "zdt1",
    "zdt1_true_front",
    "bnh",
    "bnh_feasible",
    "bnh_decision_constraints_torch",
    "dtlz2",
    "nd_front",
    "hypervolume",
    "igd",
    "sobol_sample",
    "run_mobo",
    "run_nsga2_pymoo",
]

DEV_EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "DevelopmentExample"


def zdt1(X: np.ndarray) -> np.ndarray:
    """ZDT1 objectives (minimize). ``X`` shape ``(n, d)``."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[1]
    f1 = X[:, 0]
    g = 1.0 + (9.0 / (n - 1)) * np.sum(X[:, 1:], axis=1)
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return np.column_stack([f1, f2])


def zdt1_true_front(n_points: int = 100) -> np.ndarray:
    f1 = np.linspace(0.0, 1.0, n_points)
    f2 = 1.0 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def bnh(X: np.ndarray) -> np.ndarray:
    """BNH objectives (minimize), unconstrained evaluation."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    x1, x2 = X[:, 0], X[:, 1]
    f1 = 4.0 * x1**2 + 4.0 * x2**2
    f2 = (x1 - 5.0) ** 2 + (x2 - 5.0) ** 2
    return np.column_stack([f1, f2])


def bnh_feasible(X: np.ndarray) -> np.ndarray:
    """Boolean mask for BNH analytical constraints."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    x1, x2 = X[:, 0], X[:, 1]
    g1 = (x1 - 5.0) ** 2 + x2**2 - 25.0  # <= 0
    g2 = 7.7 - ((x1 - 8.0) ** 2 + (x2 + 3.0) ** 2)  # <= 0  (original g2 >= 7.7)
    return (g1 <= 0) & (g2 <= 0)


def bnh_decision_constraints_torch():
    """BNH analytical constraints as ``g(X) <= 0`` callables (TopasMOO convention).

    ``MOBOOptimizer`` negates these for BoTorch's ``callable(x) >= 0`` API.
    Callables accept a 1-D decision vector (intra-point).
    """

    # Indexing the last axis works for both 1-D intra-point vectors and batched
    # (..., d) tensors/arrays, so no dimensionality branch is needed.
    def g1(X):
        x1, x2 = X[..., 0], X[..., 1]
        return (x1 - 5.0) ** 2 + x2**2 - 25.0

    def g2(X):
        x1, x2 = X[..., 0], X[..., 1]
        return 7.7 - ((x1 - 8.0) ** 2 + (x2 + 3.0) ** 2)

    return [g1, g2]


def dtlz2(X: np.ndarray, n_obj: int = 5) -> np.ndarray:
    """DTLZ2 (minimize)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    M = n_obj
    g = np.sum((X[:, M - 1 :] - 0.5) ** 2, axis=1)
    F = np.zeros((X.shape[0], M))
    for i in range(M):
        val = 1.0 + g
        for j in range(M - i - 1):
            val = val * np.cos(0.5 * np.pi * X[:, j])
        if i > 0:
            val = val * np.sin(0.5 * np.pi * X[:, M - i - 1])
        F[:, i] = val
    return F


def nd_front(Y: np.ndarray) -> np.ndarray:
    nds = NonDominatedSorting()
    idx = nds.do(Y, only_non_dominated_front=True)
    return Y[idx]


def hypervolume(Y: np.ndarray, ref: np.ndarray) -> float:
    front = nd_front(Y)
    return float(HV(ref_point=ref)(front))


def igd(Y: np.ndarray, true_front: np.ndarray) -> float:
    return float(IGD(true_front)(nd_front(Y)))


def sobol_sample(n: int, lower: np.ndarray, upper: np.ndarray, seed: int) -> np.ndarray:
    from scipy.stats import qmc

    d = len(lower)
    engine = qmc.Sobol(d=d, scramble=True, seed=seed)
    u = engine.random(n)
    return qmc.scale(u, lower, upper)


def run_mobo(
    *,
    objective_fn,
    lower,
    upper,
    n_obj,
    n_init,
    batch_size,
    n_batches,
    seed,
    acquisition="qlognehvi",
    num_restarts=10,
    raw_samples=512,
    decision_constraints=None,
    param_names=None,
):
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    d = len(lower)
    names = param_names or [f"x{i+1}" for i in range(d)]
    tmp = tempfile.mkdtemp(prefix="mobo_bench_")
    params = {
        "ParameterNames": names,
        "LowerBounds": lower,
        "UpperBounds": upper,
        "start_point": 0.5 * (lower + upper),
        "n_generations": n_batches,
        "n_objectives": n_obj,
    }
    opt = MOBOOptimizer(
        optimization_params=params,
        BaseDirectory=tmp,
        SimulationName="bench",
        OptimizationDirectory=DEV_EXAMPLE,
        TopasLocation="testing_mode",
        Overwrite=True,
        KeepAllResults=False,
        n_init=n_init,
        batch_size=batch_size,
        seed=seed,
        acquisition=acquisition,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        objective_fn=objective_fn,
        decision_constraints=decision_constraints,
        plot_frequency=10_000,
        final_plots=None,
        # Keep the initial design a pure Sobol/rejection design: the Sobol and
        # NSGA-II baselines get no seeded point, so injecting start_point only
        # into MOBO would bias the comparison.
        include_start_point=False,
    )
    opt.SetUpDirectoryStructure()
    opt.run(n_batches=n_batches)
    return opt


def run_nsga2_pymoo(objective_fn, lower, upper, n_obj, pop_size, n_gen, seed):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    class _P(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=len(lower),
                n_obj=n_obj,
                xl=lower,
                xu=upper,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = objective_fn(np.asarray(x).reshape(1, -1))[0]

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    res = minimize(
        _P(),
        algorithm,
        get_termination("n_gen", n_gen),
        seed=seed,
        verbose=False,
    )
    return np.atleast_2d(res.X), np.atleast_2d(res.F)
