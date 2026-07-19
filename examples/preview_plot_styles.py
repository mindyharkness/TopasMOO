#!/usr/bin/env python3
"""Render every TopasMOO plotting style and visualization type.

No TOPAS or optimizer run is required. Synthetic ZDT1-style data and mock
log/history objects drive each function in ``TopasMOO.plotting``.

Outputs (default: ``<repo>/style_previews/``, gitignored):

``styles/`` — same 2D Pareto figure in ``fast`` and each ``publication`` variant
``standalone_plots/`` — one folder per plot function (2-, 3-, and 4-objective cases)
``comprehensive/`` — full ``GenerateComprehensiveVisualizations(..., final_plots="all")``
  bundles for 2- and 3-objective mock optimizers

Usage::
    python examples/preview_plot_styles.py
    python examples/preview_plot_styles.py --output-dir /tmp/topasmoo_plots
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from TopasMOO.plotting import (
    ALL_PLOT_KEYS,
    GenerateComprehensiveVisualizations,
    plot_decision_heatmap,
    plot_hypervolume_convergence,
    plot_objective_convergence,
    plot_parallel_coordinates,
    plot_parameter_convergence,
    plot_parameter_objective_correlation,
    plot_pareto_front_2d,
    plot_pareto_front_3d,
    plot_pareto_front_projections,
    plot_petal_diagram_multi,
    plot_petal_diagram_single,
    plot_population_evolution,
)
from TopasMOO.plotting.style import apply_style, available_publication_variants

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "style_previews"

STYLE_CONFIGS: list[tuple[str, str | None]] = [
    ("fast", None),
    *[("publication", variant) for variant in available_publication_variants()],
]


def zdt1_pareto_front(n_points: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Sample and reference front for a 2-objective ZDT1-style demo."""
    f1 = np.linspace(1e-6, 1.0, n_points, endpoint=False)
    f2 = 1.0 - np.sqrt(f1)
    pareto = np.column_stack([f1, f2])
    f1_ref = np.linspace(0.0, 1.0, 200)
    true_front = np.column_stack([f1_ref, 1.0 - np.sqrt(f1_ref)])
    return pareto, true_front


def synthetic_pareto_3d(n: int = 12) -> np.ndarray:
    rng = np.random.default_rng(7)
    base = zdt1_pareto_front(n)[0]
    f3 = rng.uniform(0.2, 1.0, size=n)
    return np.column_stack([base, f3])


def synthetic_pareto_4d(n: int = 15) -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.uniform(0.1, 1.0, size=(n, 4))


def synthetic_decision_vars(n_solutions: int = 10, n_params: int = 4) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.uniform(0.0, 1.0, size=(n_solutions, n_params))


def write_optimization_log(
    path: Path,
    *,
    parameter_names: list[str],
    n_iterations: int,
    n_objectives: int,
) -> None:
    """Write a minimal OptimizationLogs.txt compatible with convergence plots."""
    rng = np.random.default_rng(99)
    lines: list[str] = []
    for it in range(n_iterations):
        parts = [f"Iteration: {it}"]
        for name in parameter_names:
            parts.append(f"{name}: {rng.uniform(0.2, 0.8):.2f}")
        for obj_idx in range(1, n_objectives + 1):
            base = 10.0 - it * 0.4 - obj_idx
            parts.append(f"ObjectiveFunction_{obj_idx}: {base:.2f}")
        lines.append(", ".join(parts) + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def build_population_history(
    reference_front: np.ndarray,
    *,
    n_generations: int = 8,
    pop_size: int = 16,
) -> list[tuple[int, np.ndarray]]:
    """Synthetic population snapshots converging toward a 2D front."""
    rng = np.random.default_rng(21)
    history: list[tuple[int, np.ndarray]] = []
    for gen in range(n_generations):
        spread = max(0.05, 1.5 * (1.0 - gen / max(n_generations - 1, 1)))
        idx = rng.integers(0, len(reference_front), size=pop_size)
        noise = rng.normal(0.0, spread, size=(pop_size, 2))
        pop = reference_front[idx] + noise
        pop = np.clip(pop, 1e-6, None)
        history.append((gen, pop))
    return history


def build_hypervolume_history(n_generations: int = 12) -> list[float]:
    """Monotonic synthetic hypervolume curve."""
    return [0.15 + 0.06 * g + 0.002 * g * g for g in range(n_generations)]


@dataclass
class MockOptimizer:
    """Minimal optimizer-shaped object for comprehensive visualization."""

    pareto_objectives: np.ndarray
    pareto_decision_vars: np.ndarray
    parameter_names: list[str]
    log_file: Path
    hypervolume_history: list[float]
    population_history: list[tuple[int, np.ndarray]]

    @property
    def n_objectives(self) -> int:
        return int(self.pareto_objectives.shape[1])

    @property
    def ParetoObjectives(self) -> np.ndarray:
        return self.pareto_objectives

    @property
    def ParetoDecisionVars(self) -> np.ndarray:
        return self.pareto_decision_vars

    @property
    def ParameterNames(self) -> list[str]:
        return self.parameter_names

    @property
    def _LogFileLoc(self) -> str:
        return str(self.log_file)

    @property
    def HypervolumeHistory(self) -> list[float]:
        return self.hypervolume_history

    @property
    def PopulationHistory(self) -> list[tuple[int, np.ndarray]]:
        return self.population_history


def _style_label(style: str, variant: str | None) -> str:
    if variant is None:
        return style
    return f"{style}_{variant}"


def render_style_gallery(out_dir: Path, pareto_2d: np.ndarray, true_front: np.ndarray) -> list[Path]:
    """Same Pareto plot under every TopasMOO matplotlib style."""
    saved: list[Path] = []
    styles_root = out_dir / "styles"
    for style, variant in STYLE_CONFIGS:
        label = _style_label(style, variant)
        dest = styles_root / label
        dest.mkdir(parents=True, exist_ok=True)
        apply_style(style, variant=variant)
        plot_pareto_front_2d(
            pareto_2d,
            dest / "pareto_front_zdt1",
            true_front=true_front,
            title=f"TopasMOO style: {label}",
            xlabel="f1",
            ylabel="f2",
        )
        saved.append(dest)
    return saved


def render_standalone_plots(out_dir: Path) -> list[Path]:
    """Call each plotting function directly with synthetic inputs."""
    apply_style("publication", variant="clean")
    root = out_dir / "standalone_plots"
    created: list[Path] = []

    pareto_2d, true_front = zdt1_pareto_front()
    pareto_3d = synthetic_pareto_3d()
    pareto_4d = synthetic_pareto_4d()
    decision = synthetic_decision_vars(n_solutions=len(pareto_2d), n_params=4)
    param_names = ["x1", "x2", "x3", "x4"]

    two_obj = root / "2_objectives"
    two_obj.mkdir(parents=True, exist_ok=True)

    plot_pareto_front_2d(
        pareto_2d, two_obj / "pareto_front_2d", true_front=true_front, show_knee_point=True
    )
    plot_parallel_coordinates(
        pareto_2d, two_obj / "parallel_coordinates", objective_names=["f1", "f2"]
    )
    plot_petal_diagram_single(
        pareto_2d[0], two_obj / "petal_diagram_single", objective_names=["f1", "f2"]
    )
    plot_parameter_objective_correlation(
        decision[:, :2],
        pareto_2d,
        two_obj / "parameter_objective_correlation",
        parameter_names=param_names[:2],
        objective_names=["f1", "f2"],
    )
    plot_decision_heatmap(
        decision[:, :2],
        two_obj / "decision_heatmap",
        parameter_names=param_names[:2],
    )
    plot_population_evolution(
        build_population_history(pareto_2d),
        two_obj / "population_evolution",
    )
    plot_hypervolume_convergence(
        build_hypervolume_history(),
        two_obj / "hypervolume_convergence",
    )

    log_2 = two_obj / "OptimizationLogs.txt"
    write_optimization_log(
        log_2,
        parameter_names=param_names[:2],
        n_iterations=12,
        n_objectives=2,
    )
    plot_objective_convergence(log_2, two_obj / "objective_convergence", n_objectives=2)
    plot_parameter_convergence(
        log_2, two_obj / "parameter_convergence", parameter_names=param_names[:2]
    )
    created.append(two_obj)

    three_obj = root / "3_objectives"
    three_obj.mkdir(parents=True, exist_ok=True)
    plot_pareto_front_3d(pareto_3d, three_obj / "pareto_front_3d", show_knee_point=True)
    plot_parallel_coordinates(
        pareto_3d, three_obj / "parallel_coordinates", objective_names=["f1", "f2", "f3"]
    )
    plot_petal_diagram_multi(
        pareto_3d, three_obj / "petal_diagram_multi", max_solutions=4
    )
    log_3 = three_obj / "OptimizationLogs.txt"
    write_optimization_log(
        log_3,
        parameter_names=param_names,
        n_iterations=10,
        n_objectives=3,
    )
    plot_objective_convergence(log_3, three_obj / "objective_convergence", n_objectives=3)
    plot_parameter_convergence(
        log_3, three_obj / "parameter_convergence", parameter_names=param_names
    )
    created.append(three_obj)

    four_plus = root / "4plus_objectives"
    four_plus.mkdir(parents=True, exist_ok=True)
    plot_pareto_front_projections(pareto_4d, four_plus / "pareto_front_projections")
    plot_parallel_coordinates(
        pareto_4d,
        four_plus / "parallel_coordinates",
        objective_names=[f"f{i + 1}" for i in range(4)],
    )
    created.append(four_plus)

    return created


def build_mock_optimizer(
    *,
    n_objectives: int,
    out_subdir: Path,
    pareto: np.ndarray,
) -> MockOptimizer:
    n_solutions = len(pareto)
    n_params = 4
    decision = synthetic_decision_vars(n_solutions=n_solutions, n_params=n_params)
    param_names = [f"x{i + 1}" for i in range(n_params)]
    log_path = out_subdir / "OptimizationLogs.txt"
    write_optimization_log(
        log_path,
        parameter_names=param_names,
        n_iterations=15,
        n_objectives=n_objectives,
    )
    pop_hist = (
        build_population_history(pareto[:, :2])
        if n_objectives >= 2
        else []
    )
    return MockOptimizer(
        pareto_objectives=pareto,
        pareto_decision_vars=decision,
        parameter_names=param_names,
        log_file=log_path,
        hypervolume_history=build_hypervolume_history(),
        population_history=pop_hist,
    )


def render_comprehensive_bundles(out_dir: Path) -> list[Path]:
    """Exercise GenerateComprehensiveVisualizations with final_plots='all'."""
    apply_style("publication", variant="nature")
    root = out_dir / "comprehensive"
    created: list[Path] = []

    pareto_2d, _ = zdt1_pareto_front(24)
    opt_2 = build_mock_optimizer(
        n_objectives=2, out_subdir=root / "_mock_2obj", pareto=pareto_2d
    )
    dest_2 = root / "2_objectives_all_plots"
    GenerateComprehensiveVisualizations(opt_2, dest_2, final_plots="all")
    created.append(dest_2)

    pareto_3d = synthetic_pareto_3d(14)
    opt_3 = build_mock_optimizer(
        n_objectives=3, out_subdir=root / "_mock_3obj", pareto=pareto_3d
    )
    dest_3 = root / "3_objectives_all_plots"
    GenerateComprehensiveVisualizations(opt_3, dest_3, final_plots="all")
    created.append(dest_3)

    return created


def write_manifest(out_dir: Path) -> Path:
    """Write a short index of what was generated."""
    lines = [
        "TopasMOO plot gallery (generated by examples/preview_plot_styles.py)",
        "",
        "Styles rendered:",
    ]
    for style, variant in STYLE_CONFIGS:
        lines.append(f"  - {_style_label(style, variant)}")
    lines.extend(
        [
            "",
            "Standalone plot modules (publication / clean):",
            "  - plot_pareto_front_2d, plot_parallel_coordinates, plot_petal_diagram_single",
            "  - plot_parameter_objective_correlation, plot_decision_heatmap",
            "  - plot_population_evolution, plot_hypervolume_convergence",
            "  - plot_objective_convergence, plot_parameter_convergence",
            "  - plot_pareto_front_3d, plot_petal_diagram_multi (+ 3-objective convergence)",
            "  - plot_pareto_front_projections, plot_parallel_coordinates (4 objectives)",
            "",
            f"Comprehensive keys (final_plots='all'): {sorted(ALL_PLOT_KEYS)}",
            "",
            "Note: default final_plots are pareto, convergence,",
            "parameter_convergence, and hypervolume. Request final_plots='all'",
            "or name keys such as petal / parallel / decision_heatmap explicitly.",
            "Each save_path writes .pdf and .png.",
        ]
    )
    manifest = out_dir / "MANIFEST.txt"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Directory for all figures (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pareto_2d, true_front = zdt1_pareto_front()

    print(f"TopasMOO plot gallery → {out_dir}")
    print("  [1/3] Style previews (fast + publication variants)…")
    render_style_gallery(out_dir, pareto_2d, true_front)

    print("  [2/3] Standalone plot functions…")
    render_standalone_plots(out_dir)

    print("  [3/3] Comprehensive bundles (final_plots='all')…")
    render_comprehensive_bundles(out_dir)

    manifest = write_manifest(out_dir)
    print(f"\nDone. Index: {manifest}")
    print(textwrap.dedent(f"""\
        Browse:
          {out_dir / 'styles'} — compare matplotlib styles on the same ZDT1 front
          {out_dir / 'standalone_plots'} — each plot function in isolation
          {out_dir / 'comprehensive'} — optimizer-style end-of-run bundles
    """))


if __name__ == "__main__":
    main()
