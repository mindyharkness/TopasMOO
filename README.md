# TopasMOO

[![CI](https://github.com/mindyharkness/TopasMOO/actions/workflows/ci.yml/badge.svg)](https://github.com/mindyharkness/TopasMOO/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/topasmoo.svg)](https://pypi.org/project/topasmoo/)
[![Python versions](https://img.shields.io/pypi/pyversions/topasmoo.svg)](https://pypi.org/project/topasmoo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

TopasMOO is a Python toolkit for multi-objective optimization of TOPAS Monte Carlo radiation therapy simulations, enabling automated discovery of Pareto-optimal simulation configurations.

## Implemented Algorithms

- **NSGA-II** (`NSGAII_Optimizer`) uses non-dominated sorting and crowding
  distance and is a strong general default for two or a few objectives.
- **NSGA-III** (`NSGAIII_Optimizer`) uses reference directions to maintain
  objective-space coverage and is especially useful as the number of objectives
  grows.

Both classes share the same TOPAS workflow, constraints, checkpointing, and
visualization support. See the [algorithm guide and API reference](docsrc/index.md#algorithms)
for selection guidance and NSGA-III reference-direction configuration.

## Installation

```bash
pip install topasmoo
# or from source:
git clone https://github.com/mindyharkness/TopasMOO.git
cd TopasMOO
pip install -e .
```

For local development with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run ruff check TopasMOO tests
uv run pytest
```

**Requirements**

- Python >= 3.10, < 3.13
- A working [TOPAS](https://topas.readthedocs.io/) installation for full Monte Carlo runs (or use `testing_mode` for development and benchmarks)

## Quick Start

The optimizer expects a project directory with `GenerateTopasScripts.py` and `TopasObjectiveFunction.py`, following the TopasOpt layout. The repository’s [examples/DevelopmentExample](examples/DevelopmentExample/) folder implements the ZDT1 benchmark: TOPAS is not run, but those two files are still present so the workflow matches a real study.

From the repository root, after installing the package:

```python
from pathlib import Path

import numpy as np
from TopasMOO import NSGAII_Optimizer

opt_dir = Path("examples/DevelopmentExample")

optimization_params = {
    "ParameterNames": ["x1", "x2", "x3", "x4", "x5"],
    "UpperBounds": np.ones(5),
    "LowerBounds": np.zeros(5),
    "start_point": np.full(5, 0.5),
    "n_generations": 20,
    "n_objectives": 2,
}

optimizer = NSGAII_Optimizer(
    optimization_params=optimization_params,
    BaseDirectory=str(opt_dir),
    SimulationName="QuickStart",
    OptimizationDirectory=opt_dir,
    TopasLocation="testing_mode",
    Overwrite=True,
    pop_size=12,
    publication_variant="clean",   # or "nature" / "ieee" / "medicalphysics"
)
results = optimizer.RunOptimization()
# results.X: decision variables on the Pareto set; results.F: objective values
```

To use reference-direction-based selection (reccomended for many-objective optimizations), import `NSGAIII_Optimizer` instead.
Its default population size is derived from its generated reference directions;
see the [NSGA-III API reference](docsrc/index.md#nsgaiii_optimizer) before choosing
the number of partitions for an expensive TOPAS run.

For a full unconstrained walkthrough, plots, and validation metrics, run
`python DevelopmentExample_main.py` inside `examples/DevelopmentExample/`.
For collimator optimization with TOPAS, see
[examples/ApertureOptimization](examples/ApertureOptimization/).

## Bayesian multi-objective optimization (MOBO)

`MOBOOptimizer` is a drop-in sibling of `NSGAII_Optimizer` for expensive campaigns (roughly **100–500** evaluations, parameter count comfortably **below ~15**). It uses BoTorch `qLogNEHVI` / `qLogNParEGO` on a shared Gaussian-process backend, exposes `ask` / `tell` / `run` for stepwise cluster submission, and feeds the shared plotting utilities plus a MOBO-specific GP prediction diagnostic.

Shared with NSGA-II: minimization objectives, `g(x) <= 0` feasible constraints, the same `optimization_params` keys, `EvaluateObjectives`, and `ParetoObjectives` / `HypervolumeHistory` / `PopulationHistory` attributes. Differences to keep in mind: `n_generations` means **acquisition batches** (after `n_init`), and the reported Pareto front is the non-dominated set over **all eligible observations** (NSGA-II reports the non-dominated set of the final population only).

```python
from TopasMOO import MOBOOptimizer

optimizer = MOBOOptimizer(
    optimization_params=optimization_params,  # n_generations = acquisition batches
    BaseDirectory=...,
    SimulationName=...,
    OptimizationDirectory=...,
    TopasLocation="testing_mode",
    n_init=20,
    batch_size=2,           # or set n_parallel_jobs to match concurrent TOPAS jobs
    acquisition="auto",     # NEHVI for 2–4 objectives, ParEGO for 5+
    seed=42,
)
results = optimizer.RunOptimization()
```

The user's `start_point` is evaluated as the first point of the initial design (as with NSGA-II); pass `include_start_point=False` to opt out.

Completed MOBO runs generate `logs/FinalResults/GPPredictionCorrelation` as
PDF and PNG by default. It compares the posterior mean recorded when each
candidate was proposed against the objective value later observed, with one
panel and Pearson/Spearman correlations per objective. Initial Sobol designs
are omitted because no GP exists yet, and penalized failures are excluded. Use
`final_plots="gp_correlation"` to request only this diagnostic, or call
`TopasMOO.plotting.plot_gp_prediction_correlation` directly.

See [examples/MOBODevelopmentExample](examples/MOBODevelopmentExample/). Prefer **NSGA-II** for larger budgets or higher-dimensional search spaces.

### Constraints

`MOBOOptimizer` constrains the **decision vector**: analytical `g(x) <= 0` callables that say which TOPAS simulation parameters are allowed. They use the pymoo convention that **`g(x) <= 0` is feasible**.

```python
optimizer = MOBOOptimizer(
    optimization_params=optimization_params,
    # Feasible where x1 + x2 <= 1. Takes a 1-D decision vector, returns a scalar.
    decision_constraints=[lambda x: x[0] + x[1] - 1.0],
    ...,
)
```

Constraints are enforced *inside* acquisition optimization, so infeasible designs are never proposed and no TOPAS run is ever spent on one. Consequently:

- The initial design switches from Sobol to uniform rejection sampling inside the feasible region, since a Sobol design carries no feasibility guarantee. If `10_000 × n_init` draws fail to find enough feasible points, the run raises rather than proceeding — the signal that the feasible region is too small to hit blindly.
- `start_point` is skipped with a warning if it violates a constraint.
- `sequential=True` is rejected with qLogNEHVI and `batch_size > 1`, because BoTorch cannot supply feasible starting points on that path.

Only observations satisfying every `decision_constraints` entry are eligible for the reported Pareto front and hypervolume history. An infeasible design is therefore never returned as an optimum even if it dominates every feasible one. If nothing feasible has been found yet, the front is empty and a warning is logged.

> **`n_constraints` is NSGA-II only.** The base-class `n_constraints` parameter — where `TopasObjectiveFunction` returns `n_objectives + k` values and feasibility is measured per evaluation — applies to `NSGAII_Optimizer`, which passes it to pymoo as `n_ieq_constr`. `MOBOOptimizer` rejects `n_constraints > 0` at construction rather than silently ignoring constraints the caller believes are enforced.

### `ask` / `tell` / `run`

For stepwise evaluation (e.g. submitting each batch to a cluster) the loop is explicit:

```python
X = optimizer.ask()                       # (n_init, d) first call, then (batch_size, d)
Y = my_evaluator(X)                       # Y: (n, n_objectives), minimization space
optimizer.tell(X, Y)
```

`ask` and `run` create the run directories themselves (and, with `resume=True`,
reload the checkpoint), so `SetUpDirectoryStructure()` no longer has to be
called first. Calling it explicitly is still supported and is then a no-op.

> **API note.** The signature is `tell(X, Y, Yvar=None, failed=None)`. Feasibility is derived from `decision_constraints` applied to `X`, so no constraint values are passed in.

### Failed evaluations

The base class defaults to `on_evaluation_failure="penalize"`, which gives a crashed TOPAS run `failure_penalty` (`1e6`) for every objective so the campaign continues. NSGA-II tolerates that — dominance is scale-free — but a Bayesian optimizer does not: a single `1e6` row would define the nadir, push the hypervolume reference out by six orders of magnitude, and dominate the GP's outcome scaling.

`RunOptimization()` / `run()` therefore detect penalized rows and record them in `train_failed`. They stay in `train_X` / `train_Y` so indices and history stay aligned, but are excluded from GP training, from both reference points, and from the reported Pareto front. Driving `ask` / `tell` yourself with an evaluator that can fail? Pass `failed=` so the same quarantine applies:

```python
optimizer.tell(X, Y, C=C, failed=np.array([False, True, False]))
```

Quarantine limits the damage but does not make a failure free — the batch still spent its budget, and a run that fails often is better diagnosed than absorbed. For campaigns where a crash means the objective is genuinely undefined rather than merely bad, consider `on_evaluation_failure="raise"`.

### Known limitations

Rough edges in `MOBOOptimizer` that are understood and deliberately not worked around. None of them produce wrong results; they cost budget, speed, or search diversity.

**`on_evaluation_failure="penalize"` is a weak default for MOBO.** Failed evaluations are quarantined (above), so a crash no longer corrupts the surrogate or the hypervolume reference. But a quarantined row still consumed a batch slot and taught the model nothing: the acquisition has no memory of the failing region and can propose it again. The evaluation cache prevents a repeat TOPAS run, so the retry is cheap in wall time — but the slot is still spent on a row that gets discarded. `"raise"` is the more honest setting when a crash means the objective is undefined. The base-class default is unchanged because it is shared with `NSGAII_Optimizer`, which genuinely does tolerate penalty rows.

**qLogNParEGO restarts from identical starting points within a batch.** When `decision_constraints` are set, `_sample_feasible_initial_conditions` seeds its RNG from `seed + 17 * batch_index`, which does not vary across the `batch_size` candidates generated inside one ParEGO batch. Every candidate therefore begins its restarts from the same feasible points. The candidates still differ — each sees fresh Chebyshev weights and a growing `X_pending` — but multi-start diversity within a batch is lower than the `num_restarts` setting suggests.

## Citation

If you use TopasMOO, please cite it (placeholder entry until a DOI is available) and the TopasOpt paper:

```bibtex
@software{harkness_topasmoo_2025,
  author       = {Harkness, Mindy},
  title        = {{TopasMOO}: Multi-objective optimization for {TOPAS} {Monte} {Carlo} simulations},
  year         = {2026},
  url          = {https://github.com/mindyharkness/TopasMOO},
  note         = {Placeholder: replace with published citation when available},
}

@article{whelan_topasopt_2022,
  title   = {{TopasOpt}: {An} open-source library for optimization with {Topas} {Monte} {Carlo}},
  journal = {Medical Physics},
  author  = {Whelan, Brendan and Loo Jr, Billy W. and Wang, Jinghui and Keall, Paul},
  year    = {2022},
  publisher = {Wiley Online Library},
}
```

## License

This project is released under the [MIT License](LICENSE).

## Related Projects

- [TopasOpt](https://github.com/Image-X-Institute/TopasOpt) — single-objective optimization for TOPAS
- [TOPAS](https://topas.readthedocs.io/) — Monte Carlo simulation for medical physics
- [pymoo](https://pymoo.org/) — multi-objective optimization algorithms in Python

---

TopasMOO is intended for **multi-objective** problems (at least two objectives). For a single scalar objective, use [TopasOpt](https://github.com/Image-X-Institute/TopasOpt).
