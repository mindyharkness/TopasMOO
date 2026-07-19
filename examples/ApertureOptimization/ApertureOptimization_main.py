"""
Multi-Objective Aperture Optimization Example, adapted from TopasOpt.
This is the main script for the example, setting and driving the optimization process. Run this script to start the optimization.
"""

from pathlib import Path

import numpy as np

from TopasMOO import optimizers as tmo

BaseDirectory = str(Path(__file__).parent)
SimulationName = "ApertureOptimization_MOO"
OptimizationDirectory = Path(__file__).parent

# Set up optimization params for multi-objective optimization
optimization_params = {}
optimization_params["ParameterNames"] = [
    "UpStreamApertureRadius",
    "DownStreamApertureRadius",
    "CollimatorThickness",
]
optimization_params["UpperBounds"] = np.array([3, 3, 40])
optimization_params["LowerBounds"] = np.array([1, 1, 10])
# Starting point
optimization_params["start_point"] = np.array([1.5, 2.0, 25.0])
optimization_params["n_generations"] = 30  # Number of generations for NSGA-II
optimization_params["n_objectives"] = 2  # Two competing objectives

# Example objectives could be:
# 1. Minimize dose uniformity error (match ground truth profile)
# 2. Maximize beam efficiency (maximize particles transmitted)
# or:
# 1. Minimize off-axis dose
# 2. Maximize on-axis dose at depth

ReadMeText = """
Multi-Objective Aperture Optimization Example

This example demonstrates multi-objective optimization of a collimator design.
We simultaneously optimize for:
1. Dose profile accuracy (matching ground truth)
2. Beam efficiency (maximizing transmission through collimator)

The result will be a Pareto front showing the trade-off between these objectives.
"""

# Initialize NSGA-II optimizer
Optimizer = tmo.NSGAII_Optimizer(
    optimization_params=optimization_params,
    BaseDirectory=BaseDirectory,
    SimulationName=SimulationName,
    OptimizationDirectory=OptimizationDirectory,
    TopasLocation="~/topas39",
    ReadMeText=ReadMeText,
    Overwrite=True,
    KeepAllResults=True,
    pop_size=10,  # Population size for NSGA-II
    seed=42,  # For reproducibility
)

# Run optimization
results = Optimizer.RunOptimization()

# Print summary
print(f"\nOptimization Complete!")
print(f"Found {len(results.F)} solutions in Pareto front")
print(f"\nPareto front objective values:")
print(results.F)
print(f"\nCorresponding parameter values:")
print(results.X)
