"""Minimal demo of TopasMOO plotting with synthetic 2-objective Pareto data.

No TOPAS or optimization: samples points on the ZDT1 analytical front
(f2 = 1 - sqrt(f1)) and writes the Pareto front (with the true front and knee
point) and parallel coordinates to a temporary directory.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from TopasMOO.plotting import plot_parallel_coordinates, plot_pareto_front_2d
from TopasMOO.plotting.style import apply_style


def main() -> None:
    apply_style("publication", variant="clean")
    f1 = np.linspace(1e-6, 1.0, 48, endpoint=False)
    F = np.column_stack([f1, 1.0 - np.sqrt(f1)])
    f1_ref = np.linspace(0.0, 1.0, 200)
    true_front = np.column_stack([f1_ref, 1.0 - np.sqrt(f1_ref)])

    out = Path(tempfile.mkdtemp(prefix="topasmoo_quickstart_"))
    plot_pareto_front_2d(F, out / "pareto", true_front=true_front, objective_names=["f1", "f2"])
    plot_parallel_coordinates(F, out / "parallel", objective_names=["f1", "f2"])
    print(f"Saved plots (PDF and PNG) under: {out}")


if __name__ == "__main__":
    main()
