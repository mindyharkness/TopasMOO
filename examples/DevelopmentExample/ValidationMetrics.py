"""Concise numerical validation for the ZDT1 development example."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from pymoo.indicators.hv import HV

from TopasMOO.plotting.style import (
    ACCENT_COLOR,
    DOUBLE_COL_WIDTH,
    finalize_figure,
    format_publication_axes,
    line_width,
    marker_area,
    publication_style,
    scale_figsize,
)

IGD_LIMIT = 0.05
MAX_FRONT_ERROR_LIMIT = 0.05
DEFAULT_HYPERVOLUME_REFERENCE = (1.1, 1.1)


@dataclass(frozen=True)
class ValidationSummary:
    """Numerical evidence that an optimization recovered the ZDT1 front."""

    solution_count: int
    igd: float
    hypervolume: float
    max_front_error: float
    f1_range: tuple[float, float]
    f2_range: tuple[float, float]
    passed: bool


def compute_true_pareto_front(n_points: int = 1000) -> np.ndarray:
    """Return evenly sampled points on the analytical ZDT1 Pareto front."""
    f1 = np.linspace(0.0, 1.0, n_points)
    return np.column_stack((f1, 1.0 - np.sqrt(f1)))


def _pairwise_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    differences = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(differences**2, axis=-1))


def compute_igd(obtained_front: np.ndarray, true_front: np.ndarray) -> float:
    """Return inverted generational distance; lower values are better."""
    distances = _pairwise_euclidean(true_front, obtained_front)
    return float(np.min(distances, axis=1).mean())


def _validated_inputs(results, hypervolume_reference):
    if not hasattr(results, "F") or results.F is None:
        raise ValueError("results.F is required")

    front = np.asarray(results.F, dtype=float)
    if front.size == 0:
        raise ValueError("results.F must be non-empty")
    if front.ndim != 2 or front.shape[1] != 2:
        raise ValueError("results.F must have shape (n_solutions, 2)")
    if not np.isfinite(front).all():
        raise ValueError("results.F must contain only finite values")
    if np.any((front[:, 0] < 0.0) | (front[:, 0] > 1.0)):
        raise ValueError("ZDT1 f1 values must lie in [0, 1]")

    reference = np.asarray(hypervolume_reference, dtype=float)
    if reference.shape != (2,) or not np.isfinite(reference).all():
        raise ValueError("hypervolume reference must contain two finite values")

    return front, reference


def calculate_zdt1_validation(
    results,
    hypervolume_reference: Sequence[float] = DEFAULT_HYPERVOLUME_REFERENCE,
) -> ValidationSummary:
    """Calculate the compact set of metrics used by the public validation."""
    front, reference = _validated_inputs(results, hypervolume_reference)
    true_front = compute_true_pareto_front()
    igd = compute_igd(front, true_front)
    front_errors = np.abs(front[:, 1] - (1.0 - np.sqrt(front[:, 0])))
    max_front_error = float(front_errors.max())
    hypervolume = float(HV(ref_point=reference)(front))

    return ValidationSummary(
        solution_count=len(front),
        igd=igd,
        hypervolume=hypervolume,
        max_front_error=max_front_error,
        f1_range=(float(front[:, 0].min()), float(front[:, 0].max())),
        f2_range=(float(front[:, 1].min()), float(front[:, 1].max())),
        passed=igd <= IGD_LIMIT and max_front_error <= MAX_FRONT_ERROR_LIMIT,
    )


def _write_report(
    summary: ValidationSummary,
    output_path: Path,
    reference: np.ndarray,
) -> None:
    status = "PASS" if summary.passed else "FAIL"
    report = (
        "ZDT1 numerical validation\n"
        f"Status: {status}\n"
        f"Pareto solutions: {summary.solution_count}\n"
        f"IGD: {summary.igd:.6f} (limit: {IGD_LIMIT:.6f})\n"
        f"Hypervolume: {summary.hypervolume:.6f} "
        f"(reference: [{reference[0]:.1f}, {reference[1]:.1f}])\n"
        f"Maximum front error: {summary.max_front_error:.6f} "
        f"(limit: {MAX_FRONT_ERROR_LIMIT:.6f})\n"
        f"f1 range: [{summary.f1_range[0]:.6f}, {summary.f1_range[1]:.6f}]\n"
        f"f2 range: [{summary.f2_range[0]:.6f}, {summary.f2_range[1]:.6f}]\n\n"
        "IGD measures coverage of the analytical front; lower is better.\n"
        "Maximum front error measures convergence to "
        "f2 = 1 - sqrt(f1); lower is better.\n"
    )
    output_path.write_text(report, encoding="utf-8")


def _plot_summary(
    front: np.ndarray,
    summary: ValidationSummary,
    reference: np.ndarray,
    save_path: Path,
) -> None:
    true_front = compute_true_pareto_front()
    with publication_style("publication"):
        fig, (front_ax, metrics_ax) = plt.subplots(
            1,
            2,
            figsize=scale_figsize(DOUBLE_COL_WIDTH, 3.5),
            gridspec_kw={"width_ratios": [1.45, 1.0]},
        )
        front_ax.plot(
            true_front[:, 0],
            true_front[:, 1],
            color="#333333",
            linewidth=line_width(),
            label="Analytical front",
        )
        front_ax.scatter(
            front[:, 0],
            front[:, 1],
            color=ACCENT_COLOR,
            s=marker_area(0.8),
            label="Obtained solutions",
            zorder=3,
        )
        front_ax.set(
            xlabel=r"$f_1$",
            ylabel=r"$f_2$",
            xlim=(-0.03, 1.03),
            ylim=(-0.03, 1.08),
            title="Pareto-front comparison",
        )
        format_publication_axes(front_ax)
        front_ax.legend(frameon=False)

        metrics_ax.set_axis_off()
        status = "PASS" if summary.passed else "FAIL"
        status_color = "#287D3C" if summary.passed else ACCENT_COLOR
        metrics_ax.text(
            0.0,
            0.95,
            "Numerical summary",
            transform=metrics_ax.transAxes,
            fontsize="large",
            fontweight="bold",
            va="top",
        )
        lines = [
            ("Pareto solutions", f"{summary.solution_count}"),
            ("IGD", f"{summary.igd:.6f}"),
            (
                f"Hypervolume\nref = [{reference[0]:.1f}, {reference[1]:.1f}]",
                f"{summary.hypervolume:.6f}",
            ),
            ("Maximum front error", f"{summary.max_front_error:.6f}"),
        ]
        y = 0.76
        for label, value in lines:
            metrics_ax.text(
                0.0, y, label, transform=metrics_ax.transAxes, va="top"
            )
            metrics_ax.text(
                1.0,
                y,
                value,
                transform=metrics_ax.transAxes,
                va="top",
                ha="right",
                fontweight="bold",
            )
            y -= 0.15
        metrics_ax.text(
            0.0,
            0.15,
            status,
            transform=metrics_ax.transAxes,
            color=status_color,
            fontsize="x-large",
            fontweight="bold",
        )
        metrics_ax.text(
            0.0,
            0.01,
            "Lower IGD/front error is better;\nhigher hypervolume is better.",
            transform=metrics_ax.transAxes,
            color="#555555",
            fontsize="small",
            va="bottom",
        )

        fig.suptitle("ZDT1 numerical validation")
        finalize_figure(fig, save_path)


def generate_zdt1_validation(
    results,
    output_dir,
    *,
    hypervolume_reference: Sequence[float] = DEFAULT_HYPERVOLUME_REFERENCE,
) -> ValidationSummary:
    """Write the public ZDT1 validation artifacts and return their metrics."""
    front, reference = _validated_inputs(results, hypervolume_reference)
    summary = calculate_zdt1_validation(results, reference)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_summary(front, summary, reference, output_dir / "zdt1_validation")
    _write_report(summary, output_dir / "zdt1_validation.txt", reference)
    return summary
