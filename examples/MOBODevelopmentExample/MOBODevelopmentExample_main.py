"""Short MOBO campaign on ZDT1 (testing_mode).

Production TOPAS runs need a larger ``n_init`` / batch budget; this example is
kept small so it finishes in a reasonable time.
"""

from pathlib import Path

import numpy as np

from examples import ValidationMetrics as vm
from TopasMOO import MOBOOptimizer

BaseDirectory = str(Path(__file__).parent)
SimulationName = "MOBODevelopmentExample_Run"
OptimizationDirectory = Path(__file__).parent

optimization_params = {
    "ParameterNames": ["x1", "x2", "x3", "x4", "x5"],
    "UpperBounds": np.ones(5),
    "LowerBounds": np.zeros(5),
    "start_point": np.full(5, 0.5),
    # n_generations = number of acquisition batches after the Sobol design
    "n_generations": 50,
    "n_objectives": 2,
}

ReadMeText = """
MOBO (BoTorch) development example on ZDT1 in testing_mode.

Install the optional extra first:
  uv sync --extra mobo

Choose MOBO when the evaluation budget is roughly below 500 simulations and
the parameter count is comfortably below ~15. Prefer NSGA-II for larger
budgets or higher-dimensional search spaces.
"""

optimizer = MOBOOptimizer(
    optimization_params=optimization_params,
    BaseDirectory=BaseDirectory,
    SimulationName=SimulationName,
    OptimizationDirectory=OptimizationDirectory,
    TopasLocation="testing_mode",
    ReadMeText=ReadMeText,
    Overwrite=True,
    KeepAllResults=False,
    n_init=25,
    batch_size=2,
    seed=42,
    acquisition="auto",
    num_restarts=5,
    raw_samples=128,
)

print("Starting MOBO optimization...")
results = optimizer.RunOptimization()
print(f"Pareto solutions: {len(results.F)}")
print(f"Final hypervolume (history): {optimizer.HypervolumeHistory[-1]:.6f}")

output_dir = Path(BaseDirectory) / SimulationName / "validation"
summary = vm.generate_zdt1_validation(results, output_dir)

status = "PASS" if summary.passed else "FAIL"
print(f"\nZDT1 validation: {status}")
print(f"  Pareto solutions: {summary.solution_count}")
print(f"  IGD: {summary.igd:.6f}")
print(f"  Hypervolume: {summary.hypervolume:.6f}")
print(f"  Maximum front error: {summary.max_front_error:.6f}")
print(f"  Outputs: {output_dir}")
