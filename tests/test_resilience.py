"""Tests for failure handling, checkpoint/resume, the evaluation cache,
constraint support, and the hardened TOPAS runner."""
from __future__ import annotations

import json
import logging
import os
import shlex
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from TopasMOO.exceptions import InvalidParameterError, ObjectiveFunctionError, TopasExecutionError
from TopasMOO.optimizers import NSGAII_Optimizer, TopasProblem


@pytest.fixture(autouse=True)
def _suppress_expected_logging():
    """Silence the WARNING/ERROR logs these tests deliberately trigger.

    Scoped to this module via ``autouse``: the previous global disable level is
    captured and restored so logging is not suppressed for the rest of the
    pytest run (other test files).
    """
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


def _write_minimal_project(opt_dir: Path, script_name: str = "MinimalSimulation") -> None:
    (opt_dir / "GenerateTopasScripts.py").write_text(
        "\n".join(
            [
                "def GenerateTopasScripts(BaseDirectory, iteration, **variable_dict):",
                "    script = [",
                "        '# minimal script',",
                "        f\"# x1 = {variable_dict['x1']}\",",
                "        f\"# x2 = {variable_dict['x2']}\",",
                "    ]",
                f"    return [script], ['{script_name}']",
                "",
            ]
        )
    )
    (opt_dir / "TopasObjectiveFunction.py").write_text(
        "\n".join(
            [
                "def TopasObjectiveFunction(ResultsLocation, iteration):",
                "    return [0.1, 0.2]",
                "",
            ]
        )
    )


def _make(base_dir: Path, name: str, opt_dir: Path | None = None, **kw) -> NSGAII_Optimizer:
    opt_dir = opt_dir or base_dir
    params = {
        "ParameterNames": ["x1", "x2"],
        "UpperBounds": np.array([1.0, 1.0]),
        "LowerBounds": np.array([0.0, 0.0]),
        "start_point": np.array([0.5, 0.5]),
        "n_generations": kw.pop("n_generations", 2),
        "n_objectives": 2,
    }
    defaults = dict(
        BaseDirectory=str(base_dir),
        SimulationName=name,
        OptimizationDirectory=opt_dir,
        TopasLocation="testing_mode",
        pop_size=4,
        seed=1,
        plot_frequency=10**9,
        final_plots=[],
    )
    defaults.setdefault("Overwrite", True)
    defaults.update(kw)
    return NSGAII_Optimizer(optimization_params=params, **defaults)


def _count_topas_runs(optimizer):
    """Wrap _run_topas_model to count how many real TOPAS (emulator) invocations occur."""
    counter = {"n": 0}
    original = optimizer._run_topas_model

    def counting():
        counter["n"] += 1
        return original()

    optimizer._run_topas_model = counting
    return counter


# --------------------------------------------------------------------------- #
# Rec 1: failure handling
# --------------------------------------------------------------------------- #


def test_penalize_topas_execution_failure(tmp_path):
    _write_minimal_project(tmp_path)
    opt = _make(tmp_path, "PenalizeExec")
    opt.SetUpDirectoryStructure()

    def boom():
        raise TopasExecutionError("simulated non-zero exit")

    opt._run_topas_model = boom
    of = opt.EvaluateObjectives(np.array([0.5, 0.5]))
    assert np.allclose(of, opt.failure_penalty)
    assert opt._n_failed_evaluations == 1


def test_penalize_non_finite_objective(tmp_path):
    _write_minimal_project(tmp_path)
    opt = _make(tmp_path, "PenalizeNaN")
    opt.SetUpDirectoryStructure()
    opt.TopasObjectiveFunction = lambda loc, it: [float("nan"), 0.2]
    of = opt.EvaluateObjectives(np.array([0.5, 0.5]))
    assert np.allclose(of, opt.failure_penalty)
    assert opt._n_failed_evaluations == 1


def test_penalize_objective_exception(tmp_path):
    _write_minimal_project(tmp_path)
    opt = _make(tmp_path, "PenalizeExc")
    opt.SetUpDirectoryStructure()

    def raising(loc, it):
        raise FileNotFoundError("could not read TOPAS output")

    opt.TopasObjectiveFunction = raising
    of = opt.EvaluateObjectives(np.array([0.5, 0.5]))
    assert np.allclose(of, opt.failure_penalty)


def test_raise_mode_propagates_failure(tmp_path):
    _write_minimal_project(tmp_path)
    opt = _make(tmp_path, "RaiseMode", on_evaluation_failure="raise")
    opt.SetUpDirectoryStructure()
    opt.TopasObjectiveFunction = lambda loc, it: [np.inf, 0.2]
    with pytest.raises(ObjectiveFunctionError):
        opt.EvaluateObjectives(np.array([0.5, 0.5]))


def test_contract_violation_always_raises_even_when_penalizing(tmp_path):
    _write_minimal_project(tmp_path)
    opt = _make(tmp_path, "ContractBug")  # default on_evaluation_failure="penalize"
    opt.SetUpDirectoryStructure()
    opt.TopasObjectiveFunction = lambda loc, it: [[0.1, 0.2]]  # 2-D: contract bug
    with pytest.raises(ObjectiveFunctionError):
        opt.EvaluateObjectives(np.array([0.5, 0.5]))


def test_invalid_on_evaluation_failure_rejected(tmp_path):
    _write_minimal_project(tmp_path)
    with pytest.raises(InvalidParameterError):
        _make(tmp_path, "BadPolicy", on_evaluation_failure="explode")


@pytest.mark.parametrize(
    "bad_freq",
    [
        0,
        -1,
        0.5,
        "often",
        2.9,   # would silently truncate to 2 under int() coercion
        True,  # bool is an int subclass; int(True) == 1 would slip through
        None,
    ],
)
def test_nonpositive_or_invalid_plot_frequency_rejected(tmp_path, bad_freq):
    _write_minimal_project(tmp_path)
    with pytest.raises(InvalidParameterError, match="plot_frequency"):
        _make(tmp_path, "BadPlotFreq", plot_frequency=bad_freq)


def test_integral_plot_frequency_accepted(tmp_path):
    _write_minimal_project(tmp_path)
    opt = _make(tmp_path, "GoodPlotFreq", plot_frequency=np.int64(5))
    assert opt.plot_frequency == 5
    assert isinstance(opt.plot_frequency, int)


@pytest.mark.parametrize("bad_penalty", [float("nan"), float("inf"), 0.0, -1.0])
def test_non_finite_or_nonpositive_failure_penalty_rejected(tmp_path, bad_penalty):
    # A non-finite penalty breaks pymoo dominance/hypervolume; 0/negative would
    # make a failed evaluation look as good as a real one, defeating the fallback.
    _write_minimal_project(tmp_path)
    with pytest.raises(InvalidParameterError):
        _make(tmp_path, "BadPenalty", failure_penalty=bad_penalty)


# --------------------------------------------------------------------------- #
# Rec 2: evaluation cache + resume
# --------------------------------------------------------------------------- #


def test_evaluate_objectives_rejects_population_matrix(tmp_path):
    # A (pop_size, n_var) matrix would otherwise be flattened into the cache key
    # and only its first row used by _create_variable_dictionary / script gen.
    _write_minimal_project(tmp_path)
    opt = _make(tmp_path, "PopMatrix")
    opt.SetUpDirectoryStructure()
    population = np.array([[0.1, 0.2], [0.3, 0.4]])  # 2 designs, n_var=2
    with pytest.raises(InvalidParameterError):
        opt.EvaluateObjectives(population)


def test_evaluate_objectives_rejects_wrong_length_vector(tmp_path):
    _write_minimal_project(tmp_path)
    opt = _make(tmp_path, "WrongLen")
    opt.SetUpDirectoryStructure()
    with pytest.raises(InvalidParameterError):
        opt.EvaluateObjectives(np.array([0.1, 0.2, 0.3]))  # n_var is 2


def test_eval_cache_skips_duplicate_simulation(tmp_path):
    _write_minimal_project(tmp_path)
    opt = _make(tmp_path, "Cache")
    opt.SetUpDirectoryStructure()
    counter = _count_topas_runs(opt)
    opt.EvaluateObjectives(np.array([0.3, 0.4]))
    opt.EvaluateObjectives(np.array([0.3, 0.4]))  # identical -> cache hit
    assert counter["n"] == 1
    assert os.path.isfile(opt._eval_cache_loc)


def test_resume_restores_evaluation_index(tmp_path):
    _write_minimal_project(tmp_path)
    first = _make(tmp_path, "EvalIdx", n_generations=2)
    first.RunOptimization()
    assert first.evaluation_index > 0
    assert os.path.isfile(first._run_state_loc)
    saved_next = first.evaluation_index

    second = _make(
        tmp_path, "EvalIdx", n_generations=2, Overwrite=False, resume=True
    )
    second.SetUpDirectoryStructure()
    assert second.evaluation_index == saved_next


def test_resume_restores_evaluation_index_from_logs_without_run_state(tmp_path):
    _write_minimal_project(tmp_path)
    first = _make(tmp_path, "EvalIdxLog", n_generations=2)
    first.RunOptimization()
    os.remove(first._run_state_loc)

    second = _make(
        tmp_path, "EvalIdxLog", n_generations=2, Overwrite=False, resume=True
    )
    second.SetUpDirectoryStructure()
    assert second.evaluation_index == first.evaluation_index


def test_official_pareto_front_matches_result_and_uses_dedicated_log(tmp_path):
    _write_minimal_project(tmp_path)
    opt = _make(tmp_path, "ParetoSplit", n_generations=2, plot_frequency=1)
    res = opt.RunOptimization()

    assert os.path.isfile(opt._ParetoLogFileLoc)
    assert np.allclose(opt.ParetoObjectives, np.atleast_2d(res.F))
    # Mid-run monitoring must not overwrite the official final front file.
    running = Path(opt._ParetoRunningLogFileLoc)
    assert running.is_file()
    official = Path(opt._ParetoLogFileLoc).read_text(encoding="utf-8")
    # Official file is rewritten at the end from res.F; running file is separate.
    assert official  # non-empty
    assert running.read_text(encoding="utf-8")  # also non-empty


def test_resume_seeds_history_from_cache_for_running_front(tmp_path):
    # Without replaying the cache into the in-memory history, the mid-run front
    # after a resume would cover post-resume evaluations only.
    _write_minimal_project(tmp_path)
    first = _make(tmp_path, "SeedHistory", n_generations=2)
    first.RunOptimization()
    n_cached = len(first._eval_cache)
    assert n_cached > 0

    second = _make(
        tmp_path, "SeedHistory", n_generations=2, Overwrite=False, resume=True
    )
    second.SetUpDirectoryStructure()
    assert len(second.AllObjectiveFunctionValues) == n_cached
    assert len(second.AllDecisionVariables) == n_cached
    # Parallel lists: _update_pareto_front only attaches decision variables when
    # the two line up, so a running front computed now carries them through.
    second._update_pareto_front()
    assert second.ParetoDecisionVars is not None
    assert len(second.ParetoObjectives) > 0
    assert Path(second._ParetoRunningLogFileLoc).is_file()


def test_resume_discards_previous_final_pareto_front(tmp_path):
    _write_minimal_project(tmp_path)
    first = _make(tmp_path, "StaleFront", n_generations=2)
    first.RunOptimization()
    official = Path(first._ParetoLogFileLoc)
    assert official.is_file()

    second = _make(
        tmp_path, "StaleFront", n_generations=2, Overwrite=False, resume=True
    )
    second.SetUpDirectoryStructure()
    # The previous run's official front must not survive into the resumed run.
    assert not official.exists()

    second.RunOptimization()
    assert official.is_file()  # rewritten once the resumed run completes


def test_resume_without_previous_final_front_is_not_an_error(tmp_path):
    # A run that crashed before finishing never wrote ParetoFront.txt.
    _write_minimal_project(tmp_path)
    opt = _make(tmp_path, "NoFront", n_generations=2, Overwrite=False, resume=True)
    opt.SetUpDirectoryStructure()  # must not raise
    assert not Path(opt._ParetoLogFileLoc).exists()


def test_resume_reuses_cache_without_rerunning_topas(tmp_path):
    _write_minimal_project(tmp_path)
    first = _make(tmp_path, "Resume", n_generations=3)
    first.RunOptimization()
    assert os.path.isfile(first._eval_cache_loc)

    second = _make(
        tmp_path, "Resume", n_generations=3, Overwrite=False, resume=True
    )
    counter = _count_topas_runs(second)
    second.RunOptimization()
    assert len(second._eval_cache) > 0
    assert counter["n"] == 0  # every design served from the cache


def test_state_checkpoint_written(tmp_path):
    _write_minimal_project(tmp_path)
    opt = _make(tmp_path, "Ckpt", n_generations=2)
    opt.RunOptimization()
    assert os.path.isfile(opt._checkpoint_loc)


def test_partial_resume_continues_without_repeating_completed_generations(tmp_path):
    _write_minimal_project(tmp_path)
    first = _make(tmp_path, "Partial", n_generations=4, eliminate_duplicates=False)
    runs = {"n": 0}
    original = first._run_topas_model

    def interrupt():
        runs["n"] += 1
        if runs["n"] > 8:  # pop_size=4 -> generations 1-2 done; crash entering gen 3
            raise KeyboardInterrupt("simulated crash")
        return original()

    first._run_topas_model = interrupt
    with pytest.raises(KeyboardInterrupt):
        first.RunOptimization()
    assert os.path.isfile(first._checkpoint_loc)
    assert len(first._eval_cache) > 0

    second = _make(
        tmp_path,
        "Partial",
        n_generations=4,
        Overwrite=False,
        resume=True,
        eliminate_duplicates=False,
    )
    counter = _count_topas_runs(second)
    res = second.RunOptimization()
    assert len(np.atleast_2d(res.F)) > 0
    # Only the remaining generations run; completed generations are not redone
    # (a full fresh run would be pop_size * n_generations = 16 evaluations).
    assert 0 < counter["n"] < 16


def test_resume_without_checkpoint_falls_back_to_cache(tmp_path):
    _write_minimal_project(tmp_path)
    first = _make(tmp_path, "FallbackCache", n_generations=3, eliminate_duplicates=False)
    first.RunOptimization()
    os.remove(first._checkpoint_loc)  # simulate a missing/corrupt state checkpoint

    second = _make(
        tmp_path,
        "FallbackCache",
        n_generations=3,
        Overwrite=False,
        resume=True,
        eliminate_duplicates=False,
    )
    counter = _count_topas_runs(second)
    second.RunOptimization()
    # The GA restarts from scratch, but every design is served from the cache.
    assert counter["n"] == 0


def test_load_eval_cache_skips_incompatible_records(tmp_path):
    # Records from a run with a different parameter count or objective/constraint
    # count must not be loaded: later slicing (raw[:n_objectives]) would misbehave.
    _write_minimal_project(tmp_path)
    opt = _make(tmp_path, "IncompatCache")  # n_var=2, n_objectives=2, n_constraints=0
    opt.SetUpDirectoryStructure()
    with open(opt._eval_cache_loc, "w") as f:
        f.write(json.dumps({"x": [0.1, 0.2], "raw": [0.5, 0.6]}) + "\n")  # compatible
        f.write(json.dumps({"x": [0.1, 0.2, 0.3], "raw": [0.5, 0.6]}) + "\n")  # x len
        f.write(json.dumps({"x": [0.4, 0.5], "raw": [0.5, 0.6, 0.7]}) + "\n")  # raw len

    opt._eval_cache = {}
    opt._load_eval_cache()
    assert list(opt._eval_cache) == [(0.1, 0.2)]
    assert np.allclose(opt._eval_cache[(0.1, 0.2)], [0.5, 0.6])


# --------------------------------------------------------------------------- #
# Rec 7: constraints, eliminate_duplicates, ShellScriptHeader
# --------------------------------------------------------------------------- #


def test_constraints_split_and_wired_into_problem(tmp_path):
    _write_minimal_project(tmp_path)
    opt = _make(tmp_path, "Constrained", n_constraints=1)
    opt.SetUpDirectoryStructure()
    opt.TopasObjectiveFunction = lambda loc, it: [0.1, 0.2, -0.3]
    f = opt.EvaluateObjectives(np.array([0.5, 0.5]))
    assert len(f) == 2 and np.allclose(f, [0.1, 0.2])
    assert np.allclose(opt._last_constraint_values, [-0.3])

    problem = TopasProblem(opt)
    assert problem.n_constr == 1


def test_constraint_count_enforced_in_return_length(tmp_path):
    _write_minimal_project(tmp_path)
    opt = _make(tmp_path, "ConstrLen", n_constraints=1)
    opt.SetUpDirectoryStructure()
    opt.TopasObjectiveFunction = lambda loc, it: [0.1, 0.2]  # missing the constraint
    with pytest.raises(ObjectiveFunctionError):
        opt.EvaluateObjectives(np.array([0.5, 0.5]))


def test_negative_n_constraints_rejected(tmp_path):
    _write_minimal_project(tmp_path)
    with pytest.raises(InvalidParameterError):
        _make(tmp_path, "BadConstr", n_constraints=-1)


def test_eliminate_duplicates_default_true(tmp_path):
    _write_minimal_project(tmp_path)
    opt = _make(tmp_path, "Dedup")
    assert opt.eliminate_duplicates is True


def test_runner_quotes_paths_and_keeps_g4_env_with_custom_header(tmp_path):
    _write_minimal_project(tmp_path)
    opt = _make(
        tmp_path,
        "Header",
        G4dataLocation="/data dir/g4",
        ShellScriptHeader="module load topas",
    )
    opt.SetUpDirectoryStructure()
    opt.EvaluateObjectives(np.array([0.5, 0.5]))
    script = Path(opt.ShellScriptLocation).read_text()
    # Geant4 export is always present and the path with a space is quoted.
    assert "TOPAS_G4_DATA_DIR='/data dir/g4'" in script
    # the custom header augments rather than replacing the environment.
    assert "module load topas" in script


def test_runner_expands_tilde_in_g4_data_dir(tmp_path):
    # ``~`` must be expanded in Python: shlex.quote would otherwise emit a
    # literal '~/G4Data' that bash does not expand, pointing TOPAS at a bogus dir.
    _write_minimal_project(tmp_path)
    opt = _make(tmp_path, "Tilde", G4dataLocation="~/G4Data")
    opt.SetUpDirectoryStructure()
    opt.EvaluateObjectives(np.array([0.5, 0.5]))
    script = Path(opt.ShellScriptLocation).read_text()
    expected = os.path.expanduser("~/G4Data")
    assert "~" not in opt.G4dataLocation
    assert f"export TOPAS_G4_DATA_DIR={shlex.quote(expected)}\n" in script
    assert "'~/G4Data'" not in script


# --------------------------------------------------------------------------- #
# R6: per-path module loading (no sys.modules name collisions)
# --------------------------------------------------------------------------- #


def test_two_projects_load_distinct_generate_scripts(tmp_path):
    dir_a = tmp_path / "projA"
    dir_b = tmp_path / "projB"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_minimal_project(dir_a, script_name="AAA")
    _write_minimal_project(dir_b, script_name="BBB")

    base = tmp_path / "runs"
    base.mkdir()
    opt_a = _make(base, "A", opt_dir=dir_a)
    opt_b = _make(base, "B", opt_dir=dir_b)

    _, names_a = opt_a.TopasScriptGenerator(".", 0, x1=0.5, x2=0.5)
    _, names_b = opt_b.TopasScriptGenerator(".", 0, x1=0.5, x2=0.5)
    assert names_a == ["AAA"]
    assert names_b == ["BBB"]


def test_user_script_can_import_sibling_helper_module(tmp_path):
    # A user GenerateTopasScripts.py that imports a helper module from the same
    # optimization directory must work: the directory is added to sys.path while
    # the module executes.
    opt_dir = tmp_path / "proj"
    opt_dir.mkdir()
    (opt_dir / "helper.py").write_text("SCRIPT_NAME = 'FromHelper'\n")
    (opt_dir / "GenerateTopasScripts.py").write_text(
        "\n".join(
            [
                "from helper import SCRIPT_NAME",
                "def GenerateTopasScripts(BaseDirectory, iteration, **variable_dict):",
                "    return [['# minimal']], [SCRIPT_NAME]",
                "",
            ]
        )
    )
    (opt_dir / "TopasObjectiveFunction.py").write_text(
        "def TopasObjectiveFunction(ResultsLocation, iteration):\n    return [0.1, 0.2]\n"
    )

    base = tmp_path / "runs"
    base.mkdir()
    opt = _make(base, "Sibling", opt_dir=opt_dir)
    _, names = opt.TopasScriptGenerator(".", 0, x1=0.5, x2=0.5)
    assert names == ["FromHelper"]
