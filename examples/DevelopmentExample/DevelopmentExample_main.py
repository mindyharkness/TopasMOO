"""Run TopasMOO against the known-solution ZDT1 benchmark."""

from pathlib import Path

import numpy as np

from examples import ValidationMetrics as vm
from TopasMOO import optimizers as tmo

BaseDirectory = str(Path(__file__).parent)
SimulationName = "DevelopmentExample_MOO"
OptimizationDirectory = Path(__file__).parent

# ZDT1 benchmark problem - standard multi-objective test with known Pareto front
optimization_params = {}
optimization_params["ParameterNames"] = ["x1", "x2", "x3", "x4", "x5"]
optimization_params["UpperBounds"] = np.array([1, 1, 1, 1, 1])
optimization_params["LowerBounds"] = np.array([0, 0, 0, 0, 0])
optimization_params["start_point"] = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
optimization_params["n_generations"] = 50
optimization_params["n_objectives"] = 2

ReadMeText = """
ZDT1 Benchmark Example for TopasMOO

This example demonstrates multi-objective optimization using the ZDT1 benchmark problem.
ZDT1 is a standard test function with a known Pareto front, making it ideal for validating
that the optimizer is working correctly.

Problem characteristics:
- 5 decision variables (x1, x2, x3, x4, x5) in range [0, 1]
- 2 objectives to minimize
- Known Pareto front: f2 = 1 - sqrt(f1) for f1 in [0, 1]

This example is useful for:
- Learning the TopasMOO API
- Testing your installation
- Validating optimizer performance against known solutions
- Understanding multi-objective optimization concepts
- Developing new objective functions before running expensive TOPAS simulations

Reference: Zitzler, E., Deb, K., & Thiele, L. (2000).
Comparison of multiobjective evolutionary algorithms: Empirical results.
"""

Optimizer = tmo.NSGAII_Optimizer(
    optimization_params=optimization_params,
    BaseDirectory=BaseDirectory,
    SimulationName=SimulationName,
    OptimizationDirectory=OptimizationDirectory,
    TopasLocation="testing_mode",
    ReadMeText=ReadMeText,
    Overwrite=True,
    KeepAllResults=False,
    pop_size=20,
    seed=42,
)

print("Starting optimization...")
results = Optimizer.RunOptimization()

output_dir = Path(BaseDirectory) / SimulationName / "validation"
summary = vm.generate_zdt1_validation(results, output_dir)

status = "PASS" if summary.passed else "FAIL"
print(f"\nZDT1 validation: {status}")
print(f"  Pareto solutions: {summary.solution_count}")
print(f"  IGD: {summary.igd:.6f}")
print(f"  Hypervolume: {summary.hypervolume:.6f}")
print(f"  Maximum front error: {summary.max_front_error:.6f}")
print(f"  Outputs: {output_dir}")
