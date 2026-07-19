# Changelog

All notable changes to TopasMOO will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

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
- `CONTRIBUTING.md` with the actual `uv` + `ruff` + `pytest` workflow.

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

- **Breaking:** `TopasMOO.utilities` module deleted. The deprecation
  wrappers (`PlotParetoFront`, `PlotConvergenceMultiObjective`,
  `PlotParameterConvergence`, `setup_publication_style`) were emitting
  `FutureWarning` since 0.1.x. Use the canonical functions in
  `TopasMOO.plotting` or `TopasMOO.plotting.style.apply_style` instead.
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
