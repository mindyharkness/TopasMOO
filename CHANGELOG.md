# Changelog

All notable changes to TopasMOO will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-07-30

First pass at implementing multi-objective Bayesian optimization (MOBO) using
BoTorch, offered alongside NSGA-II from a shared base class.

### Added

- **`MOBOOptimizer`.** Bayesian multi-objective optimizer (new module
  `TopasMOO/mobo.py`, exported as `TopasMOO.MOBOOptimizer`) built on BoTorch
  `qLogNEHVI` / `qLogNParEGO` over a `ModelListGP` of `SingleTaskGP`s.
  Kept conventions as similar to `NSGAII_Optimizer` as possible: same base-class constructor kwargs,
  `RunOptimization()` entry point, minimization and `g(x) <= 0` conventions,
  and the same `ParetoObjectives` / `HypervolumeHistory` / `PopulationHistory`
  attributes the plotting utilities read, so no plotting changes are needed at the moment.
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
  and hypervolume history. `MOBOOptimizer` rejects the base-class
  `n_constraints > 0` at construction rather than silently ignoring
  constraints the caller believes are enforced.
- **MOBO checkpoint / resume.** State is written to `logs/MOBOCheckpoint.npz`
  (plus a `.meta.json` sidecar) and validated against the current problem on
  load. A new `_restore_algorithm_state()` hook on `TopasMOOBaseClass` gives
  resume a single trigger, fired from `SetUpDirectoryStructure()`.
- `hypervolume_reference_point()` in `TopasMOO.metrics`, exported at the top
  level. Both optimizers now share one reference-point margin formula (the
  observed nadir pushed out by 10% of the observed span), so hypervolume
  histories are comparable across algorithms.
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

### Changed

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
  contradicts its own result. NSGA-II's behaviour is unchanged (it carries
  penalized and infeasible designs in its population by design).
- `plot_hypervolume_convergence()` x-axis label is now "Generation / batch",
  reflecting that the history holds one entry per algorithm step for either
  optimizer.

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
- `OptimizationSettings.json` (jsonpickle snapshot) is opt-in via
  `dump_optimization_settings=True`; off by default because resume does
  not use it.
- QUICKSTART.md documents the files a run produces under `logs/` and how to
  resume an interrupted run.
- Visualization docs (`docsrc/visualization.md`) describe the bundled
  Matplotlib styles only; the optional `scienceplots` extra was removed
  from the `examples` optional dependency group.
- `jsonpickle` moved from a required dependency to a new `settings-dump`
  extra, since nothing outside the opt-in `dump_optimization_settings`
  snapshot uses it. Install with `pip install TopasMOO[settings-dump]` if
  you enable that flag; `_copy_self` raises a message naming the extra
  when it is missing. A default install now pulls one fewer package.
- **Breaking:** American spelling throughout the optimizer API: module
  `Optimisers.py` / `optimisers.py` → `optimizers.py`; class
  `NSGAII_Optimiser` → `NSGAII_Optimizer`; methods/params such as
  `RunOptimisation` → `RunOptimization`, `optimisation_params` →
  `optimization_params`, `OptimisationDirectory` → `OptimizationDirectory`,
  and log/settings filenames (`OptimisationLogs.txt` →
  `OptimizationLogs.txt`, `OptimisationSettings.json` →
  `OptimizationSettings.json`). Recommended import:
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
