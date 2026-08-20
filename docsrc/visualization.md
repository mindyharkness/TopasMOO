# Plotting / Visualization

Publication-quality visualizations for multi-objective optimization in TopasMOO.

## Features

- Bundled matplotlib style sheets (`fast` and `publication` variants) — no
  scienceplots or LaTeX install required
- Dual PDF + PNG output (default raster DPI 600)
- Pareto fronts, parallel coordinates, petal diagrams, convergence,
  hypervolume, decision–objective views, and MOBO GP diagnostics
- One-call final suite via `GenerateComprehensiveVisualizations`

## Quick Start

```python
from TopasMOO.plotting import (
    plot_pareto_front_2d,
    plot_parallel_coordinates,
)
from TopasMOO.plotting.style import apply_style

apply_style("publication", variant="clean")

plot_pareto_front_2d(
    pareto_objectives,
    "pareto_front",
    true_front=true_pareto_front,
    show_knee_point=True,
)

plot_parallel_coordinates(
    pareto_objectives,
    "parallel",
    objective_names=["Dose Error", "Efficiency"],
)
```

Save paths are **bases without extension**; both `.pdf` and `.png` are written.

For a no-TOPAS demo of the core plots:

```bash
python examples/quickstart.py
```

## Visualization Types

### Pareto fronts

- `plot_pareto_front_2d` — 2-objective scatter, optional true-front overlay and knee point
- `plot_pareto_front_3d` — 3-objective scatter
- `plot_pareto_front_projections` — pairwise 2D panels for 4+ objectives

### Trade-offs and solution comparison

- `plot_parallel_coordinates` — normalized objectives on parallel axes
- `plot_petal_diagram_single` / `plot_petal_diagram_multi` — Nightingale roses
  (**require ≥3 objectives**)

### Decision space

- `plot_decision_heatmap` — normalized parameter heatmap + boxplots
- `plot_parameter_objective_correlation` — parameter × objective scatter grid

### Progress

- `plot_objective_convergence` / `plot_parameter_convergence` — from optimization logs
- `plot_hypervolume_convergence` — hypervolume vs generation
- `plot_population_evolution` — generation overlays (first two objectives)

### MOBO diagnostics

- `plot_gp_prediction_correlation` — one observed-versus-predicted panel per
  objective, with a perfect-prediction line and Pearson/Spearman correlations

The GP diagnostic uses the posterior mean recorded when `MOBOOptimizer.ask()`
proposes each candidate, before the objective is evaluated. Initial-design
rows have no fitted GP and are omitted, as are penalized evaluation failures.
This makes the figure a prospective model check rather than an optimistic fit
of the final GP to its own training data.

```python
from TopasMOO.plotting import plot_gp_prediction_correlation

plot_gp_prediction_correlation(
    optimizer.train_Y,
    optimizer.gp_prediction_history,
    "gp_prediction_correlation",
    objective_names=["Dose error", "Delivery time"],
    valid_mask=~optimizer.train_failed,
)
```

## Default final suite

`NSGAII_Optimizer.RunOptimization()` and `NSGAIII_Optimizer.RunOptimization()`
write figures under
`{BaseDirectory}/{SimulationName}/logs/FinalResults/` using
`GenerateComprehensiveVisualizations`.

**Default keys** (`final_plots=None` or `"default"`):

- `pareto`
- `convergence`
- `parameter_convergence`
- `hypervolume` (skipped quietly when history is unavailable)

MOBO runs also generate `GPPredictionCorrelation.{pdf,png}` by default once
prospective predictions are available. NSGA-II output is unchanged.

Pass `final_plots="all"`, a single key such as `final_plots="pareto"`, or an
explicit set for parallel coordinates, decision heatmaps, population evolution,
petal diagrams, parameter correlations, and GP correlation plots. The explicit
key for the MOBO diagnostic is `final_plots="gp_correlation"`.

```python
from TopasMOO.plotting import GenerateComprehensiveVisualizations

GenerateComprehensiveVisualizations(
    optimizer,
    save_dir="custom_plots",
    final_plots="all",  # or {"pareto", "parallel", "decision_heatmap"}
)
```

## Styling

Two base styles, with four `publication` variants:

- `fast` — cleaned-up matplotlib defaults for in-loop monitoring
- `publication`:
  - `clean` (default) — subtle grid, sans-serif, colorblind-safe palette
  - `nature` — bold sans-serif, high contrast
  - `ieee` — Computer Modern mathtext serif, boxed axes, dense ticks
  - `medicalphysics` — large sans-serif text, journal single-column authoring size

Optimizer defaults: intermediate plots use `fast`; final plots use
`publication` / `clean`. Change the variant with `publication_variant=` on the
optimizer constructor, or:

```python
from TopasMOO.plotting.style import apply_style

apply_style("publication", variant="ieee")
```

Individual `plot_*` functions call `apply_style()` themselves using the active
session style.

## API Reference

See docstrings under `TopasMOO/plotting/` (`pareto.py`, `parallel.py`,
`petal.py`, `correlation.py`, `gp_correlation.py`, `convergence.py`,
`comprehensive.py`, `style.py`).
