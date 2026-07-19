# TopasMOO

[![CI](https://github.com/mindyharkness/TopasMOO/actions/workflows/ci.yml/badge.svg)](https://github.com/mindyharkness/TopasMOO/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/topasmoo.svg)](https://pypi.org/project/topasmoo/)
[![Python versions](https://img.shields.io/pypi/pyversions/topasmoo.svg)](https://pypi.org/project/topasmoo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

TopasMOO is a Python toolkit for multi-objective optimization of TOPAS Monte Carlo radiation therapy simulations, enabling automated discovery of Pareto-optimal simulation configurations.

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

For a full walkthrough, plots, and validation metrics, run `python DevelopmentExample_main.py` inside `examples/DevelopmentExample/`. For collimator optimization with TOPAS, see [examples/ApertureOptimization](examples/ApertureOptimization/).

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
