"""Dose objectives and a measured constraint, without reference data."""

from pathlib import Path

import numpy as np
from topas2numpy import BinnedResult

# Demonstration settings, to be checked with adequate Monte Carlo statistics.
OFF_AXIS_RADIUS_MM = 20.0
MAX_OFF_AXIS_FRACTION = 0.80


def TopasObjectiveFunction(ResultsLocation, iteration):
    """Return [off-axis peak dose, -peak dose, off-axis/peak - limit].

    Both doses are scorer sums in pGy for the fixed source-history and phase-space
    reuse settings in GenerateTopasScripts.py. Off-axis means voxel centers at
    least OFF_AXIS_RADIUS_MM from the beam axis, across all phantom depths.
    The last value is a measured constraint: g <= 0 is feasible.
    """
    result = BinnedResult(str(Path(ResultsLocation) / f"WaterTank_itt_{iteration}.bin"))
    dose = np.asarray(result.data["Sum"], dtype=float)

    # Use pGy rather than tiny Gy values so MOBO can standardize the objectives.
    dose = dose * 1e12

    # topas2numpy coordinates start at the box edge, in cm. Center them on
    # the beam axis and convert to mm; the phantom is centered at x=y=0.
    coordinates = []
    for dim in result.dimensions[:2]:
        if dim.unit != "cm":
            raise ValueError("Expected scorer coordinates in cm.")
        coordinates.append(10 * (dim.get_bin_centers() - dim.n_bins * dim.bin_width / 2))
    x, y = coordinates
    off_axis = np.hypot(x[:, None], y[None, :]) >= OFF_AXIS_RADIUS_MM
    if not off_axis.any() or off_axis.all():
        raise ValueError("The scorer must contain both on-axis and off-axis voxels.")
    if not np.isfinite(dose).all():
        raise ValueError("Scored dose contains NaN or inf values.")
    peak_dose = float(dose.max())
    if peak_dose <= 0:
        raise ValueError("No dose was scored; off-axis/peak is undefined.")
    off_axis_peak = float(dose[off_axis].max())
    return [off_axis_peak, -peak_dose, off_axis_peak / peak_dose - MAX_OFF_AXIS_FRACTION]
