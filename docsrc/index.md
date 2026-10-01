# TopasMOO Documentation

Multi-objective optimization for TOPAS Monte Carlo simulations.

## Quick Navigation

| I want to...          | Go to                                   |
| --------------------- | --------------------------------------- |
| Get started quickly   | [Quick Start](#quick-start)             |
| Create visualizations | [Visualization Guide](visualization.md) |
| Understand concepts   | [Core Concepts](#core-concepts)         |
| See API details       | [API Reference](#api-reference)         |
| Understand MOBO              | [MOBO Details](#mobo)                   |

## Introduction

TopasMOO extends [TopasOpt](https://github.com/Image-X-Institute/TopasOpt) for **multi-objective optimization** with 2+ objective functions.

### When to Use TopasMOO vs TopasOpt

| Scenario                                | Use          |
| --------------------------------------- | ------------ |
| Single clear objective                  | TopasOpt     |
| Can easily combine metrics into weighted score | TopasOpt     |
| Multiple competing objectives           | **TopasMOO** |
| Need to understand trade-offs           | **TopasMOO** |
| Want Pareto-optimal solutions           | **TopasMOO** |

## Installation

```bash
pip install topasmoo
```

**Requirements:** Python >= 3.10. TOPAS MC is required for real simulations, but `testing_mode` works without TOPAS.

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

In multi-objective optimization, there is rarely a single "best" solution. Instead, you get a **Pareto front** of equally good solutions, where improving one objective comes at the cost of another objective.

#### Pareto Dominance

Solution A is said to "dominate" solution B if:

- A is better or equal in ALL objectives, AND
- A is strictly better in AT LEAST ONE objective

#### Pareto Front

The set of non-dominated solutions. Each represents a different trade-off between objectives.

### Algorithms

#### NSGA-II

Non-dominated Sorting Genetic Algorithm II, the most widely used multi-objective evolutionary algorithm. Implemented in TopasMOO using the pymoo package.

**Features**:

- Fast non-dominated sorting
- Crowding distance for diversity
- Elitist selection

**Parameters**:

- `pop_size`: Population size (default: 20)


#### NSGA-III

Non-dominated Sorting Genetic Algorithm III uses reference directions instead of crowding distance to preserve diversity across objective space. It is reccomended over NSGA-II for many-objective problems (3+). Implemented in TopasMOO using the pymoo package.

**Features**:

- Reference-direction-based survival selection
- Better control of objective-space coverage as the number of objectives grows
- The same constraint handling, failure recovery, checkpointing, and plotting
  workflow as `NSGAII_Optimizer`

**Parameters**:

- `ref_dir_method`: Reference-direction generation method (default: `"das-dennis"`)
- `ref_dir_partitions`: Number of Das-Dennis partitions (default: `12`)
- `pop_size`: Population size. The default `None` uses one individual per
  generated reference direction. An explicit value must be at least the number
  of directions.

For Das-Dennis directions, `p` partitions and `M` objectives produce `C(M + p - 1, p)` directions. The defaults therefore produce 13 directions for two objectives, 91 for three objectives, and 455 for four objectives. Because each population member can require an expensive TOPAS run, choose the partition count deliberately.

#### MOBO (Multi-Objective Bayesian Optimization)

`MOBOOptimizer` uses Gaussian-process surrogates (via BoTorch) to model the objective landscape and propose candidates that are likely to improve the coverage of the Pareto front. This is most useful when each TOPAS simulation is expensive and the evaluation budget is small, as MOBO searches the parameter space more efficiently than genetic algorithms in general.

**When to use MOBO:**

- Budget is roughly **100–500** evaluations
- Parameter count is **below ~15**

**When to use NSGA-II instead:**

- Larger evaluation budgets (thousands of evaluations)
- Higher-dimensional parameter spaces
- You do not want the optional BoTorch dependency

**Key differences from NSGA-II:**

- `n_generations` means acquisition batches after an initial `n_init` Sobol (randomly initialized) design batch
- The reported Pareto front is the non-dominated set over all eligible
  observations, not just the final population
- Requires the `mobo` optional extra: `pip install topasmoo[mobo]`

#### Choosing an Algorithm

- Start with **NSGA-II** for two-objective problems with a reasonable
  evaluation budget.
- Use **NSGA-III** when reference-direction coverage matters, particularly
  for three or more objectives. It also works with two objectives but does not
  offer an advantage over NSGA-II there.
- Use **MOBO** when simulations are expensive and the budget is small
  (100–500 evaluations, fewer than ~15 parameters).

#### Custom Algorithms

Additional pymoo algorithms can be integrated by subclassing
`TopasMOOBaseClass` and implementing `RunOptimization()`. If you implement a cool new optimization algorithm, please contribute to the repo!

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

### NSGAIII_Optimizer

```python
class NSGAIII_Optimizer(TopasMOOBaseClass):
    """Multi-objective optimizer using NSGA-III reference directions.

    Constructor arguments
    ---------------------
    optimization_params, BaseDirectory, SimulationName,
    OptimizationDirectory, ReadMeText, G4dataLocation, TopasLocation,
    ShellScriptHeader, Overwrite, KeepAllResults, plot_frequency, final_plots,
    plot_style, intermediate_plot_style, publication_variant, n_constraints,
    on_evaluation_failure, failure_penalty, resume
        Shared with NSGAII_Optimizer; see the preceding API entry for their
        definitions.
    pop_size : int | None, default None
        If None, use the number of generated reference directions. An explicit
        value must be a positive integer at least as large as that number.
        ``self.pop_size`` and the underlying pymoo algorithm store the resolved
        value.
    ref_dir_partitions : int, default 12
        Number of partitions used to generate Das-Dennis reference directions.
        Direction counts grow combinatorially with this value and the objective
        count, directly affecting the derived population size.
    ref_dir_method : str, default 'das-dennis'
        Reference-direction generation method passed to pymoo.
    seed : int, optional
        Random seed for the optimization.
    verbose : bool, default False
        If True, pymoo prints generation-by-generation progress.
    eliminate_duplicates : bool, default True
        If True, NSGA-III resamples to avoid re-evaluating identical designs.

    Raises
    ------
    InvalidParameterError
        If pop_size is neither None nor a positive integer, or if an explicit
        population is smaller than the generated reference-direction count.
    """
```

Example configuration:

```python
from TopasMOO import NSGAIII_Optimizer

optimizer = NSGAIII_Optimizer(
    optimization_params=optimization_params,
    BaseDirectory=BaseDirectory,
    SimulationName=SimulationName,
    OptimizationDirectory=OptimizationDirectory,
    TopasLocation="testing_mode",
    ref_dir_method="das-dennis",
    ref_dir_partitions=12,
    pop_size=None,  # derive from the generated directions
    seed=42,
)
```

### MOBOOptimizer

```python
class MOBOOptimizer(TopasMOOBaseClass):
    """Multi-objective Bayesian optimizer using BoTorch.

    Constructor arguments
    ---------------------
    optimization_params, BaseDirectory, SimulationName,
    OptimizationDirectory, TopasLocation, Overwrite, KeepAllResults,
    plot_frequency, final_plots, plot_style, intermediate_plot_style,
    publication_variant, n_constraints, on_evaluation_failure,
    failure_penalty, resume
        Shared with NSGAII_Optimizer; see the preceding API entry.
        ``n_generations`` is the number of acquisition batches after
        the initial design of ``n_init``.
    batch_size : int, default 1
        Candidates proposed per acquisition step.
    n_init : int, optional
        Initial design size. Default: ``max(2*d+1, 10*d)`` where d is the
        number of parameters.
    acquisition : 'auto' | 'qlognehvi' | 'qlognparego', default 'auto'
        Acquisition function. ``'auto'`` selects NEHVI for 2–4 objectives,
        ParEGO for 5+.
    num_restarts : int, default 10
        Restarts for acquisition optimization.
    raw_samples : int, default 512
        Raw samples for acquisition optimization.
    sequential : bool, default False
        qLogNEHVI candidate optimization mode. qLogNParEGO always generates
        candidates one at a time.
    seed : int, optional
        RNG seed for Sobol init, MC sampler, and acquisition restarts.
    ref_point : array-like, optional
        User-specified reference point in minimization space.
    objective_fn : callable, optional
        ``(X: ndarray) -> Y: ndarray`` for synthetic/benchmark loops that
        bypass TOPAS. Returns objectives then constraints, matching the
        standard callback convention.
    include_start_point : bool, default True
        If True, ``start_point`` replaces the first row of the initial design.
        Skipped with a warning if it violates a ``decision_constraints``
        callable.
    decision_constraints : list[callable], optional
        Analytical constraints on the input space (``g(x) <= 0`` feasible).
        Enforced during acquisition optimization so infeasible candidates
        are never proposed.
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

## Constraints

All three optimizers accept `n_constraints=k`. The objective callback returns
`n_objectives + k` values: minimized objectives first, then measured constraint
values. **A constraint is feasible when g(x) <= 0.**

```python
def TopasObjectiveFunction(results_directory, iteration):
    return [
        objective_1,
        objective_2,
        measured_leakage - maximum_leakage,
        minimum_transmission - measured_transmission,
    ]
```

### How each optimizer handles constraints

**NSGA-II / NSGA-III** pass constraint values directly to pymoo's `out["G"]`,
which uses them in tournament selection and survival.

**MOBO** fits a separate GP for each measured constraint and uses BoTorch's
outcome constraints in both qLogNEHVI and qLogNParEGO. It may evaluate
infeasible designs while learning the feasibility boundary as these observations
still train the GPs. Reported feasibility uses observed values, not a
probability threshold. Until a feasible successful observation exists, the
Pareto front is empty and hypervolume is zero.

> **Note: scale measured constraints to order one for MOBO.** MOBO passes raw
> constraint values to BoTorch, which smooths the feasibility boundary over a
> fixed width (`eta = 1e-3`). A constraint whose values are much smaller than
> that, such as a leakage difference in Gy (~1e-12), looks about 50% feasible
> everywhere, so the acquisition effectively ignores it. Very large values give
> a hard step with vanishing gradients. Express each constraint in units where
> typical values fall roughly between 0.01 and 10, for example relative to its
> limit: `measured_leakage / maximum_leakage - 1`. NSGA-II/III only compare
> constraint values, so their units do not matter.

### Analytical decision constraints (MOBO only)

Parameter `LowerBounds` / `UpperBounds` restrict each input independently in
all algorithms. MOBO additionally accepts `decision_constraints` to express
relationships between inputs that must hold **before** simulation. These use
the same `g <= 0` convention:

```python
optimizer = MOBOOptimizer(
    optimization_params=optimization_params,
    # Feasible where x1 + x2 <= 1. Takes a 1-D decision vector, returns a scalar.
    decision_constraints=[lambda x: x[0] + x[1] - 1.0],
    ...,
)
```

Analytical decision constraints are enforced inside acquisition optimization
and do not require a simulation to evaluate. When they are set:

- The initial design switches from Sobol to uniform rejection sampling inside
  the feasible region. If `10,000 × n_init` draws fail to find enough feasible
  points, the run raises. Tighten bounds or widen the constraints.
- `start_point` is skipped with a warning if it violates a constraint.
- `sequential=True` is rejected with qLogNEHVI and `batch_size > 1`, because
  BoTorch cannot supply feasible starting points on that path.

Only observations satisfying **both** the analytical and measured constraints are eligible for the Pareto front and hypervolume. Constraint observations are saved in MOBO checkpoints. Older unconstrained checkpoints still load for unconstrained runs; a constrained resume requires matching `n_constraints` and stored constraint observations.

## MOBO

Detailed reference for `MOBOOptimizer` features beyond basic usage.

### `ask` / `tell` / `run`

For stepwise evaluation (for example, submitting each batch to a cluster)
the loop is:

```python
X = optimizer.ask()                       # (n_init, d) first call, then (batch_size, d)
Y = my_evaluator(X)                       # (n, n_objectives + n_constraints)
optimizer.tell(X, Y)
```

`ask` and `run` create the run directories themselves (and, with `resume=True`,
reload the checkpoint), so `SetUpDirectoryStructure()` does not need to be
called first. Calling it explicitly is still supported.

The `tell` signature is `tell(X, Y, Yvar=None, failed=None)`. Both `Y` and
the optional synthetic `objective_fn(X)` return objectives followed by measured
constraints, matching `TopasObjectiveFunction`. Internally, `train_Y` contains
only objectives and `train_G` contains constraints. `Yvar`, `train_Yvar`, and
`TopasObjectiveVariances` remain **objective-only**, shaped `(n, n_objectives)`; constraint GPs infer their observation noise.

### Failed evaluations

The base class defaults to `on_evaluation_failure="penalize"`, which assigns
`failure_penalty` (`1e6`) to every objective of a crashed TOPAS run so the
campaign continues. NSGA-II tolerates this because dominance is scale-free.
For MOBO, however, a single 1e6 row would warp the GP's outcome scaling and push the hypervolume reference point out by orders of magnitude.

`RunOptimization()` / `run()` therefore detect penalized rows and record them
in `train_failed`. They stay in `train_X` / `train_Y` so indices and history
stay aligned, but are excluded from GP training, reference points, and the
reported Pareto front.

When driving `ask` / `tell` yourself with an evaluator that can fail, pass
`failed=` so the same quarantine applies:

```python
optimizer.tell(X, Y, failed=np.array([False, True, False]))
```

If fewer than two successful evaluations remain, `ask()` draws another initial sampling batch instead of fitting a GP. These batches consume the normal budget. A run raises after exhausting its budget if it still has fewer than two successful observations.

For campaigns where a crash means the objective is genuinely undefined rather than merely poorly chosen, consider `on_evaluation_failure="raise"`.

### GP prediction correlation

Completed MOBO runs generate `logs/FinalResults/GPPredictionCorrelation` as PDF and PNG by default. The figure compares the posterior mean recorded when
each candidate was proposed against the objective value later observed, with one panel and Pearson/Spearman correlations per objective. Initial Sobol designs are omitted (no GP exists yet), and penalized failures are excluded. Use `final_plots="gp_correlation"` to request only this diagnostic, or call `TopasMOO.plotting.plot_gp_prediction_correlation` directly.

### Known limitations

Known limitations of MOBO that do not produce wrong results but may cost budget,
speed, or search diversity are:

**`on_evaluation_failure="penalize"` is a weak for MOBO.** Failed
evaluations are quarantined (see above), so a crash no longer corrupts the surrogate or hypervolume reference. But a quarantined row still consumed a batch slot and taught the model nothing: the acquisition has no memory of the failing region and can propose it again. The evaluation cache prevents a repeat TOPAS run, but the slot is still spent on a discarded row.
`"raise"` is the better setting when a crash means the objective is undefined. The base-class default is unchanged because NSGA-II accepts penalty rows.

**qLogNParEGO restarts from identical starting points within a batch.** When
`decision_constraints` are set, the feasible-initial-conditions sampler seeds its RNG from `seed + 17 * batch_index`, which does not vary across the candidates generated inside one ParEGO batch. Every candidate therefore begins from the same feasible points. The candidates still differ, as each sees fresh Chebyshev weights and a growing `X_pending`, but multi-start diversity within a batch is lower than the `num_restarts` setting suggests.

## Examples

In-repo examples under `examples/`. The TOPAS example is adapted from TopasOpt.

| Example                                                       | Description                            | TOPAS Required? |
| ------------------------------------------------------------- | -------------------------------------- | --------------- |
| [quickstart.py](../examples/quickstart.py)                    | Synthetic Pareto + plotting only       | No              |
| [DevelopmentExample](../examples/DevelopmentExample/)         | ZDT1 benchmark with validation metrics | No              |
| [MOBODevelopmentExample](../examples/MOBODevelopmentExample/) | Short MOBO campaign on analytic ZDT1   | No              |
| [ApertureOptimization](../examples/ApertureOptimization/)     | Multi-objective collimator design      | Yes             |



## Additional Resources

- [pymoo documentation](https://pymoo.org/)
- [TopasOpt documentation](https://image-x-institute.github.io/TopasOpt/)
- [TOPAS documentation](https://topas.readthedocs.io/)
- Examples in `examples/` directory
