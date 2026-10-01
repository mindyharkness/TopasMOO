"""Read actual binary scorer fixtures through the public example callback."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("topas2numpy")

from examples.ApertureOptimization import TopasObjectiveFunction as objective


def _write_dose(directory, dose):
    path = Path(directory) / "WaterTank_itt_0.bin"
    Path(str(path) + "header").write_text(
        "# X in 3 bins of 1 cm\n"
        "# Y in 3 bins of 1 cm\n"
        "# Z in 2 bins of 1 cm\n"
        "# DoseToMedium ( Gy ) : Sum\n"
    )
    np.asarray(dose, dtype=np.float64).ravel(order="F").tofile(path)


@pytest.mark.parametrize("off_axis_peak", [0.1, 0.2, 0.4])
def test_dose_objectives_and_constraint(tmp_path, monkeypatch, off_axis_peak):
    monkeypatch.setattr(objective, "OFF_AXIS_RADIUS_MM", 10.0)
    monkeypatch.setattr(objective, "MAX_OFF_AXIS_FRACTION", 0.1)
    dose = np.zeros((3, 3, 2))
    dose[1, 1, 0] = 2e-12  # Gy on the beam axis, after centering scorer coordinates
    dose[0, 1, 1] = off_axis_peak * 1e-12  # boundary of the off-axis region
    _write_dose(tmp_path, dose)
    result = objective.TopasObjectiveFunction(tmp_path, 0)
    np.testing.assert_allclose(result, [off_axis_peak, -2.0, off_axis_peak / 2 - 0.1])


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_invalid_dose_is_not_a_feasible_solution(tmp_path, monkeypatch, value):
    monkeypatch.setattr(objective, "OFF_AXIS_RADIUS_MM", 10.0)
    _write_dose(tmp_path, np.full((3, 3, 2), value))
    with pytest.raises(ValueError):
        objective.TopasObjectiveFunction(tmp_path, 0)
