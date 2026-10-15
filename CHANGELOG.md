# Changelog

All notable changes to TopasMOO will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-10-15

First pass at implementing multi-objective Bayesian optimization (MOBO) using
BoTorch, offered alongside NSGA-II from a shared base class.

### Added

- **`MOBOOptimizer`.** Bayesian multi-objective optimizer (new module
  `TopasMOO/mobo.py`, exported as `TopasMOO.MOBOOptimizer`) built on BoTorch
  `qLogNEHVI` / `qLogNParEGO` over a `ModelListGP` of `SingleTaskGP`s.
  Kept conventions as similar to `NSGAII_Optimizer` as possible: same base-class constructor kwargs,
  `RunOptimization()` entry point, minimization and `g(x) <= 0` conventions,
  and the same `ParetoObjectives` / `ParetoDecisionVars` attributes and
  `evaluation_history()` the plotting utilities read.
  Intended to allow for fewer evaluations (roughly 100-500 evaluations, parameter
  count comfortably below ~15); prefer NSGA-II for larger budgets or
  higher-dimensional parameter spaces. Two differences from NSGA-II to keep in mind:
  `n_generations` is read as the number of acquisition batches after the
  initial design of `n_init`, and the reported Pareto front is the
  non-dominated set over all eligible observations rather than the final
  population.
- **Optional `mobo` extra** (`botorch`, `gpytorch`, `torch`). Importing
  `TopasMOO` never requires them: BoTorch is imported lazily on first use, and
  the resulting `ImportError` names the extra to install. This extra is
  optional and not installed by default so users don't need to install
  unnecessary dependencies to use the pymoo-based NSGA-II optimizer.
- **`ask` / `tell` / `run` API** on `MOBOOptimizer` for stepwise evaluation
  (e.g. submitting each batch to a cluster). `ask()` returns the initial design
  on the first call and `batch_size` candidates thereafter,
  `tell(X, Y, Yvar=None, failed=None)` takes observations in minimization
  space. `objective_fn` bypasses TOPAS entirely for synthetic/benchmark loops.
- **Analytical decision constraints** via `decision_constraints=[g, ...]`
  (`g(x) <= 0` feasible, the pymoo convention), enforced inside acquisition
  optimization so an infeasible design (e.g. geometry errors in TOPAS, etc) is never proposed and never costs a
  TOPAS run. With constraints set, the initial design switches from Sobol to
  uniform rejection sampling in the feasible region (raising if `10_000 *
  n_init` draws come up short), an infeasible `start_point` is skipped with a
  warning, and only feasible observations are eligible for the reported front
  and hypervolume history.
- **Measured constraints in `MOBOOptimizer`.** `n_constraints=k` now works as
  it does for NSGA-II/III: the callback (or `tell`) returns `n_objectives + k`
  values, constraints last, with `g <= 0` feasible. Each constraint gets its
  own GP and is passed to qLogNEHVI / qLogNParEGO as a BoTorch outcome
  constraint. Only observations feasible under both analytical and measured
  constraints are eligible for the front and hypervolume. Constraint
  observations (`train_G`) are checkpointed; older unconstrained checkpoints
  still load for unconstrained runs.
- **Recovery from early evaluation failures in MOBO.** With fewer than two
  successful evaluations, `ask()` logs a warning and draws the next batch as
  additional initial samples instead of fitting a GP. If the batch budget runs
  out first, `run()` raises `RuntimeError` rather than finalizing an empty
  result.
- **MOBO checkpoint / resume.** State is written to `logs/MOBOCheckpoint.npz`
  and validated against the current problem on load. A new `_restore_algorithm_state()` hook on `TopasMOOBaseClass` gives
  resume a single trigger, fired from `SetUpDirectoryStructure()`.
- **Prospective GP prediction correlation plots.** `MOBOOptimizer` records the
  posterior mean for every acquisition candidate before evaluation and carries
  that history through checkpoint/resume. The new
  `plot_gp_prediction_correlation()` public plotting function produces one
  observed-versus-predicted panel per objective with Pearson and Spearman
  correlations. Opt in with the `gp_correlation` final plot.
- `hypervolume_reference_point()` in `TopasMOO.metrics`, exported at the top
  level. MOBO and the hypervolume plots share one reference-point margin
  formula (the observed nadir pushed out by 10% of the observed span).
- `propagate_objective_variance()` in `TopasMOO.mobo`: first-order propagation
  of scorer variances through an objective transformation (`Cov[f] ≈ J Cov[z] Jᵀ`),
  feeding the optional known-noise path (`train_Yvar`, and the opt-in
  `use_mc_uncertainty` research flag that reads Monte Carlo scorer variances). This is untested on a full MC run, and is considered experimental.
- Benchmark suite under `benchmarks/`: ZDT1 (`run_zdt1.py`, 8 variables),
  unconstrained BNH (`run_bnh.py`), and constrained BNH + 5-objective DTLZ2
  acquisition timing (`run_phase3.py`), each with NSGA-II and Sobol controls
  and a `--quick` mode.
- `examples/MOBODevelopmentExample/`: a short MOBO campaign on analytic ZDT1 in
  `testing_mode`, plus README guidance on choosing MOBO vs NSGA-II.
- `tests/test_mobo.py` covering the optimizer, constraints, quarantine, and
  checkpoint round-trip.
- README section documenting MOBO, its constraints, the `ask` / `tell` / `run`
  loop, failure handling, and known limitations.
- **Objective names for plots.** Optional `optimization_params["ObjectiveNames"]`
  (one label per objective) replaces "Objective 1", "Objective 2", ... on every
  intermediate and final plot. The plot functions gain an `objective_names`
  argument, and `RunData` an `objective_names` field.
  The ApertureOptimization example uses it.
- **Warning when every evaluation fails.** The run summary in
  `OptimizationLogs.txt` and a logged warning now say so explicitly, instead of
  only reporting a Pareto front that contains no real results.

- **New plotting suite.** Two tiers: the end of every run writes only the
  Pareto front with its knee point (pair projections colored by the remaining
  objective for 3+ objectives); diagnostics are opt-in via `final_plots`:
  `pareto_context` (front over all other and infeasible designs in gray),
  `hypervolume` (hypervolume vs evaluations, comparable across runs),
  `constraints` (feasible share), `solution_table` (knee point and best design
  per objective, real values), `parallel` (parameters and objectives on
  real-unit axes), `decision` (parameter positions within bounds along the
  front), `parameter_convergence`, `correlation`, `gp_correlation`. During the
  run a single `logs/RunMonitor.png` (front + hypervolume) replaces the three
  intermediate plots and their PDFs.
- **`optimization_params["Maximize"]`**: display flags so a maximized objective
  (returned negated) is plotted positive under its own name.
- `TopasMOOBaseClass.evaluation_history()` / `MOBOOptimizer.evaluation_history()`:
  every evaluation with feasibility and failure masks, used by the plots.

### Removed

- Petal diagrams (`plot_petal_diagram_single`, `plot_petal_diagram_multi`),
  population evolution (`plot_population_evolution`), the decision heatmap
  (`plot_decision_heatmap`, replaced by `plot_decision_along_front`), the 3D
  Pareto plot (`plot_pareto_front_3d`, replaced by projections), the per-objective
  "running best" plot (`plot_objective_convergence`, which counted infeasible
  designs), and the per-generation hypervolume plot (`plot_hypervolume_convergence`,
  replaced by `plot_hypervolume_vs_evaluations`). Their `final_plots` keys
  (`petal`, `population_evolution`, `decision_heatmap`, `convergence`) are no
  longer recognized. `RunData` fields `observed_objectives`, `failed_mask`,
  `hypervolume_history`, and `population_history` are replaced by
  `evaluated_objectives`, `eligible`, and `failed`.
- The `fast` style and the `nature` publication variant, and with them the
  optimizer arguments `plot_style` and `intermediate_plot_style`. The run
  monitor and the final figures both use `publication_variant` (`clean`,
  `ieee`, or `medicalphysics`). `fast` was no faster than `clean` once the
  monitor became PNG-only; `nature` did not meet Nature's figure rules.
- `HypervolumeHistory` and `PopulationHistory` on `NSGAII_Optimizer` /
  `NSGAIII_Optimizer`. Nothing read them, and the hypervolume reference point
  included the 1e6 failure-penalty rows, so the values were meaningless after
  any failed evaluation. Use `evaluation_history()` with
  `plot_hypervolume_vs_evaluations`. `MOBOOptimizer` keeps both.

### Changed

- `calculate_knee_point` picks the solution with the smallest *worst*
  normalized objective (min-max) instead of the smallest sum, which ran to a
  corner of concave fronts (e.g. (0.004, 0.002, 1.0) on DTLZ2).
- Every style embeds TrueType (Type 42) fonts; `clean` and `ieee` PDFs used
  Type 3 fonts, which journals reject or flag.
- Figures no longer carry in-figure titles; captions carry them.
- `SetUpDirectoryStructure()` is now called automatically by
  `RunOptimization()` / `run()` / `ask()`, so it no longer has to be invoked by
  hand first; an explicit call remains supported, though.
- The `n_generations` / `n_iterations` key now documents algorithm steps,
  with the meaning defined by the subclass (NSGA-II population generations, or
  MOBO acquisition batches). A missing key raises `InvalidParameterError`
  instead of a bare `ValueError`.
- `logs/EvalCache.jsonl` records a `failed` flag per evaluation, so a resumed
  run knows which cached designs were penalties rather than inferring it from
  the values (an objective may legitimately return `failure_penalty`). Cache
  files written before this field load as not-failed.
- New `_eligible_rows()` hook on the base class: the mid-run
  `ParetoFront_Running.txt` uses the same definition of "reportable" as the
  final front, so a long MOBO run no longer shows a monitoring front that
  contradicts its own result. NSGA-II's behavior is unchanged (it carries
  penalized and infeasible designs in its population by design).
- The `medicalphysics` publication variant now authors at the journal's true
  80 mm column with 10 pt text and saves at that exact width (no tight crop).
  Plot functions take scatter marker edges from the style (`lines.markeredgewidth`),
  and `plot_pareto_front_2d` places its legend below the axes when it owns the figure.
- CI lints with `ruff check .`, which now covers `benchmarks/` as well as
  `TopasMOO/` and `tests/`; `examples/` is excluded from ruff.

### Fixed

- A constrained NSGA-II/III run in which no design is feasible no longer crashes
  at the end. pymoo returns `res.F = None`; this became a 1x1 NaN array that
  `LogParetoFrontToFile` could not reshape, so `ParetoFront.txt`, the run
  summary, and the final plots were never written. The front is now empty (as
  in `MOBOOptimizer`) and the run summary warns that no feasible design was found.
- The `medicalphysics` variant now follows the journal's figure guidelines for
  every plot: widths snap to the 80 mm or 180 mm column (double-column and
  multi-panel figures were 160 mm or wider than the page), text is never
  smaller than 10 pt as printed (petal labels were 6-7 pt), the parts of
  multi-panel figures are labeled (a), (b), ..., and titles wider than the
  column wrap instead of being cut off.
- Axis ticks are spaced for the available room (`MaxNLocator(nbins="auto")`)
  instead of up to 10 per axis, which overprinted labels on long axes (e.g.
  "8001000120014001600") and in narrow multi-panel grids.
- A relative `BaseDirectory`, `TopasLocation`, or `G4dataLocation` no longer
  makes every evaluation fail. `RunIteration.sh` runs from `TopasScripts/`, so
  relative paths missed the script/binary and every design was penalized while
  the run still reported a Pareto front. These paths are now made absolute.
- NSGA-III checkpoints work with pymoo >= 0.6.1.6, which wraps NSGA-III's
  tournament comparator in a closure that pickle cannot reference (every
  `Checkpoint.pkl` was 0 bytes). Selection behavior is unchanged.
- pymoo checkpoints are written to a temporary file and swapped in, so a failed
  write no longer truncates the previous good checkpoint.
- `OptimizationLogs.txt` and `ParetoFront.txt` record 6 significant digits
  instead of 2 / 6 fixed decimals, so small objectives (e.g. raw TOPAS doses in
  Gy) are no longer written, and plotted, as 0.
- The `ieee` variant draws underscores. It used Computer Modern (`cmr10`), whose
  legacy encoding renders `_` as a dot accent (`dose_ratio` became `dose˙ratio`);
  it now uses a Times-style serif with STIX math.
- `NSGAIII_Optimizer(ref_dir_method=...)` rejects methods other than
  `das-dennis`, `uniform`, and `incremental` with an `InvalidParameterError`.
  pymoo's `energy`, `layer-energy`, `multi-layer`, and `reduction` take a point
  count instead of partitions and failed with a `TypeError`.
- `MOBOOptimizer(seed=...)` no longer reseeds NumPy's global random generator;
  it only ever drew from its own generators.
- The ZDT1 example objective functions raise when they cannot read the design
  back from its TOPAS script, instead of silently scoring `x = [0.5] * 5`.

## [0.2.0] - 2026-07-18

First release published to PyPI. Pre-1.0 cleanup; breaking changes from
0.1.x are listed below.

### Added

- **Resilient evaluations.** `on_evaluation_failure` (`"penalize"` default,
  or `"raise"`) on the optimizer: a TOPAS run that exits non-zero, an
  objective that raises, or a non-finite objective value is logged and
  assigned `failure_penalty` (default `1e6`) so a long campaign is not lost
  to one bad design. Contract violations (wrong return type/shape/length)
  still raise.
- **Checkpoint / resume.** Every evaluation is cached at full precision in
  `logs/EvalCache.jsonl` and the NSGA-II state is pickled each generation to
  `logs/Checkpoint.pkl`. Constructing the optimizer with `resume=True`
  continues a crashed/interrupted run in the same folder without repeating
  completed TOPAS simulations.
- **Inequality constraint support** via `n_constraints`: when `> 0` the
  objective function returns `n_objectives + n_constraints` values
  (objectives first, then `g(x) <= 0` constraint values), wired into pymoo's
  `out["G"]`.
- `eliminate_duplicates` constructor arg on `NSGAII_Optimizer` (default
  `True`) so identical decision vectors are not re-simulated.
- `available_publication_variants()` exported at the top level
  (`TopasMOO.available_publication_variants`) and from
  `TopasMOO.plotting`.
- `verbose` parameter on `NSGAII_Optimizer` (default `False`) controls
  whether pymoo prints per-generation progress to stdout.
- Type hints on the public functions in `TopasMOO.io`,
  `TopasMOO.metrics`, and the `TopasMOO.plotting` subpackage.
- GitHub Actions release workflow (`release.yml`) using PyPI trusted
  publishing on `v*` tags, plus a `build` job in CI that validates the
  sdist/wheel via `twine check`.
- Three publication variants in the plotting style system: `clean`
  (SciencePlots-clean), `nature` (modern Nature/Science), `ieee`
  (Computer Modern serif). Selectable via `apply_style("publication",
  variant=...)` or the `publication_variant` constructor arg on the
  optimizer.

### Changed

- Default end-of-run plot set (`DEFAULT_FINAL_PLOTS`) is now lean:
  `pareto`, `convergence`, `parameter_convergence`, and `hypervolume`.
  Parallel coordinates, petal diagrams, decision heatmaps, correlation,
  and population-evolution plots remain available via `final_plots="all"`
  or an explicit key set.
- `final_plots="pareto"` (and other single key strings) now selects that
  one plot instead of silently generating nothing via character-set
  iteration.
- `plot_frequency` must be a positive integer; values `<= 0`, non-integral
  numbers (`2.9`), booleans, and non-numbers raise `InvalidParameterError`
  rather than being coerced or crashing on modulo.
- Official final Pareto front (`ParetoFront.txt`, end-of-run figures,
  `ParetoObjectives`) always matches `res.F` / `res.X`. Mid-run monitoring
  still uses the ND set over all evaluations and writes
  `ParetoFront_Running.txt` so the two definitions no longer overwrite the
  same file.
- Resume restores `evaluation_index` from `logs/RunState.json` (or the
  optimization log) so iteration numbering and script names continue
  instead of restarting at 0.
- Resume replays `EvalCache.jsonl` into the in-memory evaluation history, so
  the running front in `ParetoFront_Running.txt` spans the whole run instead
  of post-resume evaluations only. (Hypervolume history still covers
  post-resume generations only.)
- Resume deletes the previous run's `ParetoFront.txt` at start-up. Only the
  end-of-run path writes that file, so a resumed run that crashed would
  otherwise leave the earlier run's front looking like the current result.
- Shared `plot_pareto_front()` dispatcher selects 2D / 3D / projections
  from objective count (used by intermediate and final plot paths).
- QUICKSTART.md documents the files a run produces under `logs/` and how to
  resume an interrupted run.
- Visualization docs (`docsrc/visualization.md`) describe the bundled
  Matplotlib styles only; the optional `scienceplots` extra was removed
  from the `examples` optional dependency group.
- **Breaking:** American spelling throughout the optimizer API: module
  `Optimisers.py` / `optimisers.py` → `optimizers.py`; class
  `NSGAII_Optimiser` → `NSGAII_Optimizer`; methods/params such as
  `RunOptimisation` → `RunOptimization`, `optimisation_params` →
  `optimization_params`, `OptimisationDirectory` → `OptimizationDirectory`,
  and log filenames (`OptimisationLogs.txt` → `OptimizationLogs.txt`).
  Recommended import:
  `from TopasMOO import NSGAII_Optimizer`.
- **Breaking:** `_EmptySimulationFolder` no longer prompts via `input()`
  and no longer raises `SystemExit`. When the simulation folder is
  non-empty and `Overwrite=False`, it now raises `RuntimeError` with a
  message pointing the user at `Overwrite=True`.
- **Breaking:** Optimizer logging no longer attaches a `StreamHandler` at
  import time. Embed-style users now control logging via the standard
  library's `logging.basicConfig(...)` (or any handler configuration of
  their choice) and no longer get duplicate or hijacked log output.
- **Breaking:** `TopasMOO.io.ReadInMultiObjectiveLogFile` raises
  `MalformedOutputError` on a non-numeric value in a row instead of
  silently dropping that field. `MalformedOutputError` was already in
  the public API; it is now actually raised.
- All `print()` calls inside `optimizers.py` were converted to
  `logger.info` / `logger.warning` (no more inline ANSI color codes
  in log messages).
- Default plot style for end-of-run figures is `publication` with
  variant `clean`. Default intermediate style remains `fast`.
- User scripts (`GenerateTopasScripts.py`, `TopasObjectiveFunction.py`) are
  now imported under a name unique to their absolute path
  (`importlib.util.spec_from_file_location`) instead of a bare module name on
  `sys.path`. This fixes silently reusing the first project's scripts when
  two optimizations with different `OptimizationDirectory`s run in one process.
- The generated `RunIteration.sh` quotes all paths (`shlex.quote`), so
  `BaseDirectory`/`TopasLocation` may contain spaces. A custom
  `ShellScriptHeader` is now appended *after* the standard shebang +
  `TOPAS_G4_DATA_DIR` export (it augments the environment instead of silently
  replacing it).
- `NSGAII_Optimizer.RunOptimization` now drives pymoo with an ask/tell loop
  (equivalent results to `minimize` for a given seed) to enable
  per-generation checkpointing.

### Removed

- **Breaking:** The deprecated public wrappers in `TopasMOO.utilities`
  (`PlotParetoFront`, `PlotConvergenceMultiObjective`,
  `PlotParameterConvergence`, `setup_publication_style`) were removed; they
  had been emitting `FutureWarning` since 0.1.x. Use the canonical functions
  in `TopasMOO.plotting` or `TopasMOO.plotting.style.apply_style` instead.
  The module itself remains, but now holds only internal helpers and is not
  part of the public API.
- Sphinx, Read-the-docs, and recommonmark dev-dependencies removed.
  Docs are now Markdown-only under `docsrc/`.

### Fixed

- TOPAS log path in `_RunTopasModel` error messages now points at the
  actual log location (`{BaseDirectory}/{SimulationName}/logs/TopasLogs`).
- `_safe_savefig` no longer raises when `bbox_inches` is unsupported by
  the current backend.

## [0.1.1]

Internal pre-release; not published to PyPI.

## [0.1.0]

Initial code drop. Not published to PyPI.
