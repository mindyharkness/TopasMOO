# Plotting / Visualization

Publication-quality figures for multi-objective optimization in TopasMOO.

Every figure follows the same rules: one message per figure, no titles (captions
carry them), gray for context, color only where it encodes a value, PDF + PNG
output (600 dpi), and fonts embedded as TrueType so journals accept the PDFs.

## What a run writes

| When | File | Contents |
| --- | --- | --- |
| During the run, every `plot_frequency` evaluations | `logs/RunMonitor.png` | Current Pareto front of the feasible designs + hypervolume vs evaluations. PNG only; overwritten. |
| End of every run (default) | `logs/FinalResults/ParetoFront_Final.{pdf,png}` | Pareto front and knee point. With 3+ objectives: one panel per objective pair, colored by the remaining objective. |
| End of run, opt-in | `logs/FinalResults/…` | The diagnostics below, selected with `final_plots`. |

## Labels and maximized objectives

```python
optimization_params = {
    ...,
    "ObjectiveNames": ["Off-axis peak dose (pGy)", "Peak dose (pGy)"],
    "Maximize": [False, True],
}
```

`ObjectiveNames` labels every axis, legend, and table (default "Objective 1",
...). TopasMOO minimizes every objective, so the objective function still
returns a maximized quantity negated (`-peak_dose`); `Maximize` only changes the
display, so plots show `+peak_dose` under its own name.

## Knee point

The highlighted solution is the **knee point**: the Pareto solution whose worst
normalized objective is smallest (`TopasMOO.metrics.calculate_knee_point`). It is
the most balanced trade-off and, unlike a minimum-sum choice, does not run to a
corner of a concave front.

## Opt-in diagnostics

Select with `final_plots=` on the optimizer, or call
`GenerateComprehensiveVisualizations` on a finished optimizer (or a `RunData`).
`final_plots="all"` writes every figure the run has data for.

| Key | File | Shows |
| --- | --- | --- |
| `pareto_context` | `ParetoFront_Context` | 2-objective front over every other design (gray dots) and infeasible designs (gray ×). Axes frame the feasible designs. |
| `hypervolume` | `HypervolumeVsEvaluations` | Hypervolume of the feasible designs found after each number of evaluations (= TOPAS runs). |
| `constraints` | `ConstraintCheck` | Share of feasible designs over a moving window. Constrained runs only. |
| `solution_table` | `SolutionTable` | Knee point and the best design for each objective, with real objective and parameter values. |
| `parallel` | `ParallelCoordinates_Final` | Parameters and objectives on real-unit axes (ends = bounds / front range); gray lines, knee point highlighted. |
| `decision` | `DecisionVariables` | Where each parameter sits within its bounds, for every Pareto solution ordered along the front. |
| `parameter_convergence` | `ParameterConvergence_Final` | Parameter values over evaluations, from `OptimizationLogs.txt`. |
| `correlation` | `ParameterObjectiveCorrelation` | Each parameter vs each objective across the Pareto set. |
| `gp_correlation` | `GPPredictionCorrelation` | MOBO only: GP prediction made before each evaluation vs the observed value. |

```python
from TopasMOO.plotting import GenerateComprehensiveVisualizations

GenerateComprehensiveVisualizations(optimizer, "custom_plots", final_plots="all")
GenerateComprehensiveVisualizations(optimizer, "custom_plots",
                                    final_plots={"pareto_context", "solution_table"})
```

### Comparing runs

`plot_hypervolume_vs_evaluations` accepts a list of runs and puts them on one
reference point, so algorithms with different population or batch sizes can be
compared per TOPAS run:

```python
from TopasMOO.plotting import plot_hypervolume_vs_evaluations

runs = [opt.evaluation_history() for opt in (nsga2, mobo)]
plot_hypervolume_vs_evaluations(
    [F for F, _eligible, _failed in runs],
    "hv_comparison",
    eligible=[eligible for _F, eligible, _failed in runs],
    labels=["NSGA-II", "MOBO"],
)
```

### MOBO GP prediction correlation

The GP diagnostic uses the posterior mean recorded when `MOBOOptimizer.ask()`
proposes each candidate, before the objective is evaluated. Initial-design rows
have no fitted GP and are omitted, as are penalized evaluation failures. This
makes the figure a prospective model check rather than an optimistic fit of the
final GP to its own training data.

## Styling

Two base styles, with four `publication` variants:

- `fast` — cleaned-up matplotlib defaults for the in-run monitor
- `publication`:
  - `clean` (default) — subtle grid, sans-serif, colorblind-safe palette
  - `nature` — bold sans-serif, high contrast
  - `ieee` — Computer Modern mathtext serif, boxed axes, dense ticks
  - `medicalphysics` — follows the [Medical Physics author guidelines](https://aapm.onlinelibrary.wiley.com/hub/journal/24734209/about/author-guidelines):
    sans-serif text (Arial/Helvetica/Calibri), no bold, figures exactly 80 mm
    (single column) or 180 mm (double column; multi-panel grids snap to this)
    wide so text prints at 10 pt or larger, (a), (b), ... labels on the parts of
    multi-panel figures, 600 dpi, and fonts embedded as TrueType

Optimizer defaults: the run monitor uses `fast`; final figures use
`publication` / `clean`. Change the variant with `publication_variant=` on the
optimizer constructor, or:

```python
from TopasMOO.plotting.style import apply_style

apply_style("publication", variant="ieee")
```

Individual `plot_*` functions call `apply_style()` themselves using the active
session style.

To preview every style and figure without running an optimization:

```bash
python examples/preview_plot_styles.py --output-dir /tmp/topasmoo_plots
```

## API Reference

See docstrings under `TopasMOO/plotting/` (`pareto.py`, `progress.py`,
`table.py`, `parallel.py`, `decision.py`, `convergence.py`, `correlation.py`,
`gp_correlation.py`, `comprehensive.py`, `style.py`).
