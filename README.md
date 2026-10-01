# TopasMOO

[![CI](https://github.com/mindyharkness/TopasMOO/actions/workflows/ci.yml/badge.svg)](https://github.com/mindyharkness/TopasMOO/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/topasmoo.svg)](https://pypi.org/project/topasmoo/)
[![Python versions](https://img.shields.io/pypi/pyversions/topasmoo.svg)](https://pypi.org/project/topasmoo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Multi-objective optimization for [TOPAS](https://topas.readthedocs.io/) Monte Carlo radiation therapy simulations. TopasMOO solves optimization problems with 2+ objectives and returns Pareto optimal solutions, where one objective cannot be improved without making other objectives worse.

## Algorithms

- **NSGA-II** (`NSGAII_Optimizer`) — non-dominated sorting with crowding
  distance (implemented using pymoo). A strong general default for two or a few objectives.
- **NSGA-III** (`NSGAIII_Optimizer`) — reference-direction-based selection (implemented using pymoo).
  Especially useful as the number of objectives grows beyond two.
- **MOBO** (`MOBOOptimizer`) — Gaussian-process surrogates (implemented using BoTorch). Best
  when simulations are expensive and the evaluation budget is small.

All three share the same TOPAS workflow, constraint convention, checkpointing,
and visualization support. See the [algorithm guide and API reference](docsrc/index.md#algorithms) for selection guidance.

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

TopasMOO expects a project directory containing two files that follow the original
[TopasOpt](https://github.com/Image-X-Institute/TopasOpt) conventions: `GenerateTopasScripts.py` (builds TOPAS input files from the current parameters) and `TopasObjectiveFunction.py` (reads simulation output and returns objective values). The
[examples/DevelopmentExample](examples/DevelopmentExample/) folder shows this structure on the ZDT1 benchmark problem.

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

For reference-direction-based selection (recommended when the number of
objectives grows), import `NSGAIII_Optimizer` instead. Its default population
size is derived from its generated reference directions; see the
[NSGA-III API reference](docsrc/index.md#nsgaiii_optimizer) before choosing
the number of partitions for an expensive TOPAS run.

For a full unconstrained walkthrough, plots, and validation metrics, run
`python DevelopmentExample_main.py` inside `examples/DevelopmentExample/`.
For collimator optimization with TOPAS and a measured dose constraint, see [examples/ApertureOptimization](examples/ApertureOptimization/).

## Bayesian Multi-Objective Optimization (MOBO)

When each TOPAS simulation is computationally expensive and you can only afford a few hundred evaluations, `MOBOOptimizer` fits Gaussian-process surrogates to guide the parameter-space search instead of relying on a large population. It is a drop-in alternative to `NSGAII_Optimizer` with the same constructor keys, same minimization convention, and same plotting. MOBO works best for simulation budgets of roughly **100–500** evaluations with **fewer than ~15** parameters.

```python
from TopasMOO import MOBOOptimizer

optimizer = MOBOOptimizer(
    optimization_params=optimization_params,  # n_generations = acquisition batches
    BaseDirectory=...,
    SimulationName=...,
    OptimizationDirectory=...,
    TopasLocation="testing_mode",
    n_init=20,
    batch_size=2,
    acquisition="auto",     # NEHVI for 2–4 objectives, ParEGO for 5+
    seed=42,
)
results = optimizer.RunOptimization()
```

Two differences from NSGA-II: `n_generations` means acquisition batches
(after the `n_init` initial design), and the reported Pareto front is the
non-dominated set over all eligible observations rather than just the final population.

See [examples/MOBODevelopmentExample](examples/MOBODevelopmentExample/) for a
walkthrough, and the [full documentation](docsrc/index.md#mobo) for constraints, stepwise `ask`/`tell` usage, failure handling, and other MOBO-specific details.

## Constraints

All three optimizers accept `n_constraints=k`. The objective callback returns
`n_objectives + k` values (minimized objectives first, then constraint values). A constraint is feasible when **g(x) <= 0**.

```python
def TopasObjectiveFunction(results_directory, iteration):
    # ...
    return [
        objective_1,
        objective_2,
        measured_leakage - maximum_leakage,       # g <= 0 is feasible
        minimum_transmission - measured_transmission,
    ]
```

MOBO also supports analytical `decision_constraints` that restrict the input space before simulation, so infeasible geometries never cost a TOPAS run. See the [constraint documentation](docsrc/index.md#constraints) for details.

## Citation

If you use TopasMOO, please cite it (placeholder entry until a DOI is available) and the TopasOpt paper:

```bibtex
@software{harkness_topasmoo_2026,
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
