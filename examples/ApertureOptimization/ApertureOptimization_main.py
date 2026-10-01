"""Constrained aperture optimization using dose scores from TOPAS."""

import argparse
from pathlib import Path

import numpy as np

from TopasMOO import MOBOOptimizer, NSGAII_Optimizer, NSGAIII_Optimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["mobo", "nsga2", "nsga3"], default="mobo")
    parser.add_argument("--topas", default="~/topas39", help="TOPAS installation directory")
    parser.add_argument("--g4-data", default="~/G4Data", help="Geant4 data directory")
    parser.add_argument(
        "--overwrite", action="store_true", help="Clear this algorithm's previous results first"
    )
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    params = {
        "ParameterNames": ["UpStreamApertureRadius", "DownStreamApertureRadius", "CollimatorThickness"],
        "UpperBounds": np.array([3.0, 3.0, 40.0]),
        "LowerBounds": np.array([1.0, 1.0, 10.0]),
        "start_point": np.array([1.5, 2.0, 25.0]),
        "n_generations": 10,
        "n_objectives": 2,
    }
    optimizer_class, settings = {
        "mobo": (MOBOOptimizer, {"n_init": 15, "batch_size": 2, "acquisition": "qlognehvi"}),
        "nsga2": (NSGAII_Optimizer, {"pop_size": 20}),
        "nsga3": (NSGAIII_Optimizer, {"ref_dir_partitions": 19}),
    }[args.algorithm]
    optimizer = optimizer_class(
        optimization_params=params,
        n_constraints=1,
        BaseDirectory=directory,
        SimulationName=f"ApertureOptimization_{args.algorithm}",
        OptimizationDirectory=directory,
        TopasLocation=args.topas,
        G4dataLocation=args.g4_data,
        ReadMeText="Minimize off-axis peak dose and maximize peak dose; ratio limit is set in TopasObjectiveFunction.py.",
        Overwrite=args.overwrite,
        KeepAllResults=True,
        on_evaluation_failure="penalize",
        seed=42,
        **settings,
    )
    results = optimizer.RunOptimization()
    if results.F is None or not np.size(results.F):
        print("No feasible Pareto solutions found.")
        return
    print("Pareto objectives [off-axis peak dose (pGy), negative peak dose (pGy)]:")
    print(results.F)
    print("Parameters [upstream radius, downstream radius, full thickness] (mm):")
    print(results.X)


if __name__ == "__main__":
    main()
