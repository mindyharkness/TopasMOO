"""Tests for MOBO Gaussian-process prediction correlation plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from TopasMOO.plotting import plot_gp_prediction_correlation


def test_gp_correlation_plots_each_objective_and_saves_both_formats(tmp_path) -> None:
    observed = np.array(
        [
            [0.0, 4.0, 2.0],
            [1.0, 3.0, 1.0],
            [2.0, 2.0, 3.0],
            [3.0, 1.0, 4.0],
        ]
    )
    predicted = observed + np.array([0.1, -0.1, 0.2])

    axes = plot_gp_prediction_correlation(
        observed,
        predicted,
        tmp_path / "gp_correlation",
        objective_names=["Dose", "Time", "Error"],
    )

    assert axes.shape == (1, 3)
    assert [ax.get_title() for ax in axes.flat] == ["Dose", "Time", "Error"]
    assert "Spearman: 1.00" in axes[0, 0].texts[0].get_text()
    assert "Pearson: 1.00" in axes[0, 0].texts[0].get_text()
    assert (tmp_path / "gp_correlation.pdf").is_file()
    assert (tmp_path / "gp_correlation.png").is_file()


def test_gp_correlation_filters_initial_predictions_and_failed_rows() -> None:
    observed = np.array(
        [
            [10.0, 20.0],
            [9.0, 18.0],
            [8.0, 16.0],
            [7.0, 14.0],
            [6.0, 12.0],
        ]
    )
    predicted = np.array(
        [
            [np.nan, np.nan],
            [8.8, 18.2],
            [7.9, 16.1],
            [1000.0, 1000.0],
            [6.1, 11.8],
        ]
    )
    valid = np.array([True, True, True, False, True])

    axes = plot_gp_prediction_correlation(observed, predicted, valid_mask=valid)
    try:
        offsets = axes[0, 0].collections[0].get_offsets()
        assert len(offsets) == 3
        assert "n = 3" in axes[0, 0].texts[0].get_text()
    finally:
        plt.close(axes[0, 0].figure)


def test_gp_correlation_uses_multirow_layout_and_hides_unused_axes() -> None:
    observed = np.arange(20, dtype=float).reshape(5, 4)
    predicted = observed + 0.25

    axes = plot_gp_prediction_correlation(observed, predicted)
    try:
        assert axes.shape == (2, 3)
        assert axes[1, 0].get_title() == "Objective 4"
        assert not axes[1, 1].get_visible()
        assert not axes[1, 2].get_visible()
    finally:
        plt.close(axes[0, 0].figure)


def test_gp_correlation_marks_undefined_metrics_as_unavailable() -> None:
    observed = np.ones((3, 2))
    predicted = np.ones((3, 2))

    axes = plot_gp_prediction_correlation(observed, predicted)
    try:
        assert "Spearman: n/a" in axes[0, 0].texts[0].get_text()
        assert "Pearson: n/a" in axes[0, 0].texts[0].get_text()
    finally:
        plt.close(axes[0, 0].figure)


@pytest.mark.parametrize(
    ("observed", "predicted", "message"),
    [
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), "observed_objectives"),
        (np.ones((3, 2)), np.ones((3, 3)), "same shape"),
    ],
)
def test_gp_correlation_rejects_invalid_shapes(observed, predicted, message) -> None:
    with pytest.raises(ValueError, match=message):
        plot_gp_prediction_correlation(observed, predicted)


def test_gp_correlation_validates_names_and_mask() -> None:
    observed = np.ones((3, 2))
    predicted = np.ones((3, 2))

    with pytest.raises(ValueError, match="one label per objective"):
        plot_gp_prediction_correlation(
            observed,
            predicted,
            objective_names=["Only one"],
        )
    with pytest.raises(ValueError, match="one entry per observation"):
        plot_gp_prediction_correlation(
            observed,
            predicted,
            valid_mask=np.array([True, False]),
        )
