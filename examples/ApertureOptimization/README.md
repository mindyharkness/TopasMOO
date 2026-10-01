# Constrained aperture optimization

This TOPAS example, adapted from TopasOpt, adjusts a collimator using only the
water-tank dose scored for each candidate. It demonstrates how TopasMOO works
on a simple two-objective, one-constraint problem. No reference dataset is
needed. MOBO is the default algorithm; NSGA-II and NSGA-III use the same
objective callback.

## Objectives and measured constraint

The two objectives are to **minimize peak off-axis dose** and **maximize peak
phantom dose**. The latter is multiplied by -1 because TopasMOO
minimizes every objective.

Off-axis voxels have centers at least **20 mm** from the beam axis in the
transverse plane. Both maxima are taken over all phantom depths. A candidate
is feasible only when its off-axis peak is at most **80%** of its phantom peak:

```python
return [
    off_axis_peak,                         # minimize, pGy
    -peak_dose,                            # maximize, pGy
    off_axis_peak / peak_dose - 0.80,      # g <= 0 is feasible
]
```

This uses `n_objectives=2` and `n_constraints=1`. Edit `OFF_AXIS_RADIUS_MM` and
`MAX_OFF_AXIS_FRACTION` in `TopasObjectiveFunction.py` to change the region and
limit. The 20 mm / 80% settings are illustrative, not a validated design
specification. A three-design smoke check with these settings gave off-axis
peak ratios of approximately 0.71, 1.00, and 0.94; the 0.80 limit separates
those samples. Check its effect with sufficient simulation statistics before
interpreting the optimized designs. A zero-dose result raises an error because
the ratio would be undefined; missing or invalid scores are never replaced by
made-up objectives.

The constraint is evaluated **after TOPAS runs**. MOBO learns it with a GP and
may simulate infeasible candidates while exploring. NSGA-II/III use the measured
violation in selection. The final feasible Pareto set expresses the tradeoff
between peak output and off-axis dose within the allowed ratio.

## Allowed parameter values

| Parameter | Bounds |
|---|---|
| `UpStreamApertureRadius` | 1–3 mm |
| `DownStreamApertureRadius` | 1–3 mm |
| `CollimatorThickness` | 10–40 mm, full thickness |

These input bounds apply before simulation. They do not guarantee compliance
with the measured dose constraint. The beam travels toward negative z, so the
upstream aperture is the cone's positive-z face; TOPAS receives half the full
thickness as the cone's `HL`.

## Run

Install the example reader and, for MOBO, BoTorch:

```bash
uv sync --extra examples --extra mobo
uv run python examples/ApertureOptimization/ApertureOptimization_main.py --algorithm mobo --topas ~/topas39 --g4-data ~/G4Data
```

Use `--algorithm nsga2` or `--algorithm nsga3` for the evolutionary algorithms.
Running again with the same algorithm stops rather than replacing earlier
results; pass `--overwrite` to clear them first.
`--help` lists the options. A real TOPAS installation and its Geant4 data are
required; testing mode does not generate dose results for this example.

The driver sets 10 optimization steps: MOBO uses 15 initial evaluations followed
by 10 batches of 2; NSGA-II/III use populations of 20 over 10 generations. Results go under `ApertureOptimization_<algorithm>`:

- `TopasScripts/`: the collimator and water-tank input files for each candidate.
- `Results/`: the intermediate phase space and water-tank dose files.
- `logs/`: optimization progress and final Pareto results.

The callback reads the binary dose scorer's **Sum** in Gy using `topas2numpy`
and converts it to pGy for the objective values. This fixed unit conversion
keeps the small doses numerically suitable for MOBO.
Keep source histories, variance-reduction settings, phase-space reuse, and
scoring bins fixed across candidates so the dose objectives are comparable.
The generator uses 50 source histories per candidate, with phase-space
reuse of 200. These small source statistics are for demonstration purposes.
Increase source histories and check repeatability before drawing conclusions about feasibility near the limit.

MOBO filters infeasible evaluations from both running and final fronts. For
NSGA-II/III, the running front is an objective-space monitor and can include
infeasible points; use the final optimizer result to assess accepted solutions.
