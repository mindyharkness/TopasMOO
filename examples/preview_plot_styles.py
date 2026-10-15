#!/usr/bin/env python3
"""Render every TopasMOO plotting style and figure from synthetic data.

No TOPAS or optimizer run is required.

Outputs (default: ``<repo>/style_previews/``, gitignored):

``styles/`` — the same Pareto front in each ``publication`` variant
``suite_2_objectives/`` — ``final_plots="all"`` for a constrained 2-objective run,
  plus the in-run monitor
``suite_3_objectives/`` — the same for a 3-objective run (pair projections)

Usage::
    python examples/preview_plot_styles.py
    python examples/preview_plot_styles.py --output-dir /tmp/topasmoo_plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from TopasMOO.plotting import (
    GenerateComprehensiveVisualizations,
    RunData,
    plot_pareto_front_2d,
    plot_run_monitor,
)
from TopasMOO.plotting.progress import nondominated_eligible
from TopasMOO.plotting.style import apply_style, available_publication_variants

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "style_previews"


def write_log(path: Path, X: np.ndarray, F: np.ndarray, names: list[str]) -> None:
    """Minimal OptimizationLogs.txt, as written by the optimizers."""
    rows = []
    for i, (x, f) in enumerate(zip(X, F, strict=True)):
        cols = [f"Iteration: {i}"] + [f"{n}: {v:.6g}" for n, v in zip(names, x, strict=True)]
        cols += [f"ObjectiveFunction_{j + 1}: {v:.6g}" for j, v in enumerate(f)]
        rows.append(", ".join(cols))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def synthetic_run(out_dir: Path, n_objectives: int, n_evaluations: int = 300) -> RunData:
    """Random designs drifting toward a ZDT1 (2 obj) or DTLZ2 (3 obj) front."""
    rng = np.random.default_rng(n_objectives)
    names = [f"x{i + 1}" for i in range(4)]
    drift = np.linspace(1.0, 0.05, n_evaluations)[:, None]
    X = rng.uniform(0, 1, (n_evaluations, 4))
    X[:, 1:] *= drift  # later designs move toward the front
    g = 1 + 9 * X[:, 1:].mean(axis=1) if n_objectives == 2 else 1 + ((X[:, 2:] - 0.5) ** 2).sum(axis=1)
    if n_objectives == 2:
        F = np.c_[X[:, 0], g * (1 - np.sqrt(X[:, 0] / g))]
        eligible = F.sum(axis=1) >= 0.9  # a made-up constraint for the constraint check
    else:
        a, b = X[:, 0] * np.pi / 2, X[:, 1] * np.pi / 2
        F = g[:, None] * np.c_[np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)]
        eligible = np.ones(n_evaluations, bool)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_log(out_dir / "OptimizationLogs.txt", X, F, names)
    front = nondominated_eligible(F, eligible)
    predicted = F + rng.normal(0, 0.03, F.shape)
    predicted[:30] = np.nan  # no GP before the initial design, as in MOBO
    return RunData(
        pareto_objectives=F[front],
        n_objectives=n_objectives,
        parameter_names=names,
        pareto_decision_vars=X[front],
        evaluated_objectives=F,
        eligible=eligible,
        failed=np.zeros(n_evaluations, bool),
        n_constraints=1 if n_objectives == 2 else 0,
        lower_bounds=[0.0] * 4,
        upper_bounds=[1.0] * 4,
        objective_names=[f"$f_{j + 1}$" for j in range(n_objectives)],
        log_file=out_dir / "OptimizationLogs.txt",
        gp_prediction_history=predicted,
    )


def render(out_dir: Path) -> None:
    f1 = np.linspace(0, 1, 40, endpoint=False)
    front = np.c_[f1, 1 - np.sqrt(f1)]
    true_front = np.c_[np.linspace(0, 1, 200), 1 - np.sqrt(np.linspace(0, 1, 200))]
    for variant in available_publication_variants():
        apply_style("publication", variant=variant)
        folder = out_dir / "styles" / variant
        folder.mkdir(parents=True, exist_ok=True)
        plot_pareto_front_2d(front, folder / "pareto_front_zdt1",
                             true_front=true_front, objective_names=["$f_1$", "$f_2$"])

    apply_style("publication", variant="clean")
    for m in (2, 3):
        suite = out_dir / f"suite_{m}_objectives"
        run = synthetic_run(suite, m)
        GenerateComprehensiveVisualizations(run, suite, final_plots="all")
        plot_run_monitor(run.evaluated_objectives, suite / "RunMonitor", eligible=run.eligible,
                         objective_names=run.objective_names, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    out = parser.parse_args().output_dir.resolve()
    render(out)
    print(f"Wrote {len(list(out.rglob('*.png')))} PNGs (each with a PDF) under {out}")


if __name__ == "__main__":
    main()
