# TopasMOO Documentation

Multi-objective optimization for TOPAS Monte Carlo simulations.

## Quick Navigation

| I want to...          | Go to                                   |
| --------------------- | --------------------------------------- |
| Get started quickly   | [Quick Start](#quick-start)             |
| Create visualizations | [Visualization Guide](visualization.md) |
| Understand concepts   | [Core Concepts](#core-concepts)         |
| See API details       | [API Reference](#api-reference)         |

## Introduction

TopasMOO extends [TopasOpt](https://github.com/Image-X-Institute/TopasOpt) for **multi-objective optimization** where you need to optimize multiple competing objectives simultaneously.

### When to Use TopasMOO vs TopasOpt

| Scenario                                | Use          |
| --------------------------------------- | ------------ |
| Single clear objective                  | TopasOpt     |
| Can combine metrics into weighted score | TopasOpt     |
| Multiple competing objectives           | **TopasMOO** |
| Need to understand trade-offs           | **TopasMOO** |
| Want Pareto-optimal solutions           | **TopasMOO** |

## Installation

```bash
pip install topasmoo
```

**Requirements:** Python ≥ 3.10. TOPAS MC is required for real simulations, but `testing_mode` works without TOPAS.

## Quick Start

### 1. Basic Setup

```python
import numpy as np
from pathlib import Path
from TopasMOO import NSGAII_Optimizer

optimization_params = {
    'ParameterNames': ['aperture_size', 'beam_energy'],
    'UpperBounds': np.array([10.0, 15.0]),
    'LowerBounds': np.array([1.0, 6.0]),
    'start_point': np.array([5.0, 10.0]),
    'n_generations': 30,
    'n_objectives': 2,
}

optimizer = NSGAII_Optimizer(
    optimization_params=optimization_params,
    BaseDirectory='/path/to/results',
    SimulationName='my_optimization',
    OptimizationDirectory=Path(__file__).parent,
    TopasLocation='~/topas',
    pop_size=20,
    publication_variant='clean',  # 'clean' | 'nature' | 'ieee'
    verbose=False,                # set True to enable per-generation pymoo output
)

results = optimizer.RunOptimization()
```

### 2. Define Objective Function

Create `TopasObjectiveFunction.py`:

```python
def TopasObjectiveFunction(ResultsLocation, iteration):
    """
    Calculate multiple objectives from TOPAS results.
    MUST return a list or numpy array of objective values.
    """
    # Load TOPAS results
    from TopasOpt.utilities import WaterTankData
    data = WaterTankData(ResultsLocation, f'dose_itt_{iteration}.bin')

    # Calculate objectives (all to be minimized)
    objective1 = calculate_dose_uniformity(data)  # Lower is better
    objective2 = -calculate_dose_coverage(data)   # Negate for maximization

    return [objective1, objective2]  # MUST be a list or array
```

### 3. Generate TOPAS Scripts

Create `GenerateTopasScripts.py`:

```python
def GenerateTopasScripts(BaseDirectory, iteration, **variable_dict):
    """
    Generate TOPAS input files with current parameter values.
    """
    # Extract optimized parameters
    aperture_size = variable_dict['aperture_size']
    beam_energy = variable_dict['beam_energy']

    # Build TOPAS script
    script = []
    script.append(f'd:Ge/Aperture/HLX = {aperture_size} mm')
    script.append(f'd:So/Beam/Energy = {beam_energy} MeV')
    # ... rest of TOPAS commands ...

    return [script], ['SimulationName']
```

## Core Concepts

### Multi-Objective Optimization

In multi-objective optimization, there is rarely a single "best" solution. Instead, you get a **Pareto front** of equally good solutions.

#### Pareto Dominance

Solution A dominates solution B if:

- A is better or equal in ALL objectives, AND
- A is strictly better in AT LEAST ONE objective

#### Pareto Front

The set of non-dominated solutions. Each represents a different trade-off between objectives.

### Algorithms

#### NSGA-II (Implemented)

Non-dominated Sorting Genetic Algorithm II - the most popular multi-objective evolutionary algorithm.

**Features**:

- Fast non-dominated sorting
- Crowding distance for diversity
- Elitist selection

**Parameters**:

- `pop_size`: Population size (default: 20)


#### Custom Algorithms

Feel free to implement additional multi-objective algorithms from pymoo!

## API Reference

### NSGAII_Optimizer

```python
class NSGAII_Optimizer(TopasMOOBaseClass):
    """Multi-objective optimizer using NSGA-II.

    Constructor arguments
    ---------------------
    optimization_params : dict
        Must include 'n_objectives' (>=2), 'ParameterNames', 'start_point',
        'UpperBounds', 'LowerBounds', 'n_generations' (alias: 'n_iterations').
    BaseDirectory : str | Path
        Existing root directory under which 'SimulationName' is created.
    SimulationName : str
        Subfolder name for this optimization run.
    OptimizationDirectory : str | Path
        Directory containing 'GenerateTopasScripts.py' and 'TopasObjectiveFunction.py'.
    TopasLocation : str
        TOPAS install root, or 'testing_mode' for unit-test/no-TOPAS development.
    Overwrite : bool, default False
        If True, clear an existing 'SimulationName' folder before the run.
        With False, raises RuntimeError when the folder is non-empty.
    KeepAllResults : bool, default True
        If False, clear 'Results/' before each iteration.
    plot_frequency : int, default 10
        Number of objective *evaluations* between intermediate convergence plots
        (not generations).
    final_plots : 'default' | 'all' | iterable[str] | None
        Set of plot keys to generate at the end of optimization (see
        :func:`TopasMOO.plotting.GenerateComprehensiveVisualizations`).
    plot_style : 'fast' | 'publication', default 'publication'
        Style for the end-of-run figures.
    intermediate_plot_style : 'fast' | 'publication', default 'fast'
        Style for plots generated mid-optimization.
    publication_variant : 'clean' | 'nature' | 'ieee' | 'medicalphysics', default 'clean'
        Variant of the publication style.
    n_constraints : int, default 0
        Number of inequality constraints (g(x) <= 0 is feasible). When > 0 the
        objective function must return n_objectives + n_constraints values:
        the objectives first, then the constraint values.
    on_evaluation_failure : 'penalize' | 'raise', default 'penalize'
        How to handle a TOPAS run that exits non-zero, an objective that raises,
        or a non-finite objective value. 'penalize' logs it and assigns
        ``failure_penalty`` so the run continues; 'raise' aborts. Contract
        violations (wrong type/shape/length) always raise.
    failure_penalty : float, default 1e6
        Objective value assigned to each objective of a penalized failure. Must
        be worse (larger) than any real objective so failed designs are dominated.
    resume : bool, default False
        If True, continue a previous run in the same simulation folder: the
        evaluation cache and per-generation checkpoint are loaded so completed
        simulations are not repeated and the folder is not cleared.
    pop_size : int, default 20
        NSGA-II population size. The final front size and objective-space
        coverage are bounded by pop_size; resolving a real front well usually
        needs a larger population.
    seed : int, optional
        Random seed for the optimization.
    verbose : bool, default False
        If True, pymoo prints generation-by-generation progress.
    eliminate_duplicates : bool, default True
        If True, NSGA-II resamples to avoid re-evaluating identical designs
        (avoids wasting TOPAS runs). Note: as a stochastic GA setting it can
        occasionally alter front spread for a given seed.
    """
```

### Key Methods

#### RunOptimization()

```python
results = optimizer.RunOptimization()
```

Runs the multi-objective optimization.

**Returns**: pymoo Result object with:

- `results.X`: Pareto optimal parameter values
- `results.F`: Corresponding objective values

#### EvaluateObjectives(x)

```python
objectives = optimizer.EvaluateObjectives(parameters)
```

Evaluates objectives for given parameters. Called automatically during optimization.

## Examples

In-repo examples under `examples/`. TOPAS examples adapted from TopasOpt.

| Example                                                       | Description                            | TOPAS Required? |
| ------------------------------------------------------------- | -------------------------------------- | --------------- |
| [quickstart.py](../examples/quickstart.py)                    | Synthetic Pareto + plotting only       | No              |
| [DevelopmentExample](../examples/DevelopmentExample/)         | ZDT1 benchmark with validation metrics | No              |
| [ApertureOptimization](../examples/ApertureOptimization/)     | Multi-objective collimator design      | Yes             |



## Additional Resources

- [pymoo documentation](https://pymoo.org/)
- [TopasOpt documentation](https://image-x-institute.github.io/TopasOpt/)
- [TOPAS documentation](https://topas.readthedocs.io/)
- Examples in `examples/` directory
