# TopasMOO Quick Start Guide

## Install

```bash
pip install topasmoo
```
## Three-File Structure

Every TopasMOO optimization requires **three files** in the same directory:

### File 1: Optimization Driver Script (`*_main.py`)

```python
import tempfile
from pathlib import Path

import numpy as np

from TopasMOO import NSGAII_Optimizer

optimization_params = {
    'ParameterNames': ['x1', 'x2'],
    'UpperBounds': np.array([10, 10]),
    'LowerBounds': np.array([0, 0]),
    'start_point': np.array([5, 5]),
    'n_generations': 20,
    'n_objectives': 2,  # REQUIRED: must be >= 2
}

optimizer = NSGAII_Optimizer(
    optimization_params=optimization_params,
    BaseDirectory=tempfile.mkdtemp(),                # any existing directory works
    SimulationName='MyOptimization',
    OptimizationDirectory=Path(__file__).parent,
    TopasLocation='testing_mode',                    # use a real path for TOPAS
    Overwrite=True,
    pop_size=10,
    publication_variant='clean',                     # 'clean' | 'nature' | 'ieee' | 'medicalphysics'
)

results = optimizer.RunOptimization()
print(f"Found {len(results.F)} Pareto-optimal solutions!")
```

### File 2: Objective Function (`TopasObjectiveFunction.py`)

```python
def TopasObjectiveFunction(ResultsLocation, iteration):
    """
    Calculate objectives from TOPAS results.
    MUST return a list or numpy array.
    """
    # Load your TOPAS results here
    # from TopasOpt.utilities import WaterTankData
    # data = WaterTankData(ResultsLocation, f'dose_itt_{iteration}.bin')

    obj1 = calculate_first_objective()   # e.g., dose uniformity
    obj2 = calculate_second_objective()  # e.g., beam efficiency

    return [obj1, obj2]  # MUST be a list or array
```

### File 3: TOPAS Simulation Script Generator (`GenerateTopasScripts.py`)

```python
def GenerateTopasScripts(BaseDirectory, iteration, **variable_dict):
    """Generate TOPAS input files with current parameter values."""
    x1 = variable_dict['x1']
    x2 = variable_dict['x2']

    script = []
    script.append(f'd:Ge/MyGeometry/HLX = {x1} mm')
    script.append(f'd:So/MySource/Energy = {x2} MeV')
    # ... more TOPAS lines ...

    return [script], ['SimulationName']
```

## Run Your Optimization

```bash
python my_main.py
```

## What You Get

Everything lands under `<BaseDirectory>/<SimulationName>/`:

| Path                             | Contents                                                        |
| -------------------------------- | --------------------------------------------------------------- |
| `logs/OptimizationLogs.txt`      | One row per evaluation: iteration, parameters, objective values  |
| `logs/ParetoFront.txt`           | **Official** final front, written once the run completes         |
| `logs/ParetoFront_Running.txt`   | Mid-run front, refreshed every `plot_frequency` evaluations      |
| `logs/FinalResults/`             | End-of-run figures (see `final_plots`)                           |
| `logs/ConvergencePlot.*`         | Intermediate monitoring plots, refreshed during the run          |
| `logs/EvalCache.jsonl`           | Completed evaluations, replayed by `resume=True`                 |
| `logs/Checkpoint.pkl`            | Per-generation algorithm state, used by `resume=True`            |
| `logs/RunState.json`             | Evaluation counter, so `resume=True` continues its numbering     |
| `logs/TopasLogs/`                | Raw TOPAS stdout/stderr per iteration                            |
| `TopasScripts/`                  | Generated TOPAS input files                                      |
| `Results/`                       | TOPAS scoring output                                             |

`ParetoFront.txt` and `ParetoFront_Running.txt` answer different questions.
The running file is the non-dominated set over *every evaluation so far* — it
exists while the run is in flight so you can watch progress. The official file
is the optimizer's own final population (`res.F` / `res.X`) and matches the
end-of-run figures exactly. If a run crashes before finishing, only the running
file will be present.

## Resuming an Interrupted Run

Re-run the same script with `resume=True` (and the same `SimulationName`):

```python
optimizer = NSGAII_Optimizer(
    ...,
    Overwrite=False,
    resume=True,
)
```

Completed simulations are read from `EvalCache.jsonl` instead of re-running
TOPAS, iteration numbering continues rather than restarting at 0, and the
previous run's `ParetoFront.txt` is cleared so it cannot be mistaken for the
current result. Hypervolume history covers post-resume generations only.
