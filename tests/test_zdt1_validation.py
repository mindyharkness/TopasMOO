from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pymoo.indicators.hv import HV

from examples.DevelopmentExample.ValidationMetrics import (
    IGD_LIMIT,
    MAX_FRONT_ERROR_LIMIT,
    ValidationSummary,
    calculate_zdt1_validation,
    compute_true_pareto_front,
    generate_zdt1_validation,
)


def result_with(front):
    return SimpleNamespace(F=np.asarray(front, dtype=float))


def test_analytical_front_passes_with_expected_metrics():
    front = compute_true_pareto_front(1000)

    summary = calculate_zdt1_validation(result_with(front))

    assert isinstance(summary, ValidationSummary)
    assert summary.solution_count == 1000
    assert summary.igd == pytest.approx(0.0, abs=1e-12)
    assert summary.max_front_error == pytest.approx(0.0, abs=1e-12)
    assert summary.hypervolume == pytest.approx(
        HV(ref_point=np.array([1.1, 1.1]))(front)
    )
    assert summary.passed
    assert IGD_LIMIT == 0.05
    assert MAX_FRONT_ERROR_LIMIT == 0.05


def test_poor_front_fails_without_rejecting_points_outside_hv_reference():
    summary = calculate_zdt1_validation(result_with([[0.0, 2.0], [1.0, 2.0]]))

    assert summary.hypervolume == pytest.approx(0.0)
    assert not summary.passed


@pytest.mark.parametrize(
    ("results", "message"),
    [
        (SimpleNamespace(), "results.F"),
        (result_with([]), "non-empty"),
        (result_with([[0.0, 1.0, 2.0]]), "shape"),
        (result_with([[np.nan, 1.0]]), "finite"),
        (result_with([[-0.1, 1.0]]), r"\[0, 1\]"),
    ],
)
def test_invalid_fronts_raise_clear_errors(results, message):
    with pytest.raises(ValueError, match=message):
        calculate_zdt1_validation(results)


def test_invalid_hypervolume_reference_raises():
    with pytest.raises(ValueError, match="reference"):
        calculate_zdt1_validation(
            result_with([[0.0, 1.0]]),
            hypervolume_reference=(1.1,),
        )


def test_generator_writes_only_documented_artifacts(tmp_path):
    front = compute_true_pareto_front(30)

    summary = generate_zdt1_validation(result_with(front), tmp_path)

    assert summary.passed
    assert {path.name for path in tmp_path.iterdir()} == {
        "zdt1_validation.png",
        "zdt1_validation.pdf",
        "zdt1_validation.txt",
    }


def test_report_matches_returned_summary(tmp_path):
    summary = generate_zdt1_validation(
        result_with(compute_true_pareto_front(30)),
        tmp_path,
    )

    report = (tmp_path / "zdt1_validation.txt").read_text()
    assert f"IGD: {summary.igd:.6f}" in report
    assert f"Hypervolume: {summary.hypervolume:.6f}" in report
    assert f"Maximum front error: {summary.max_front_error:.6f}" in report
    assert "Status: PASS" in report
    assert "limit: 0.050000" in report


def test_development_example_uses_single_validation_entry_point():
    source = Path(
        "examples/DevelopmentExample/DevelopmentExample_main.py"
    ).read_text(encoding="utf-8")

    assert source.count("generate_zdt1_validation(") == 1
    assert "plot_comparison_with_true_front(" not in source
    assert "plot_decision_space(" not in source
    assert "generate_detailed_report(" not in source
    assert "generate_all_publication_plots(" not in source
