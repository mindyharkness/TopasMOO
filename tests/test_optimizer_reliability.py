from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from TopasMOO.exceptions import InvalidParameterError, ObjectiveFunctionError
from TopasMOO.optimizers import NSGAII_Optimizer


def _write_minimal_project(opt_dir: Path) -> None:
    (opt_dir / "GenerateTopasScripts.py").write_text(
        "\n".join(
            [
                "def GenerateTopasScripts(BaseDirectory, iteration, **variable_dict):",
                "    script = [",
                "        '# minimal script',",
                "        f\"# x1 = {variable_dict['x1']}\",",
                "        f\"# x2 = {variable_dict['x2']}\",",
                "    ]",
                "    return [script], ['MinimalSimulation']",
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


def _make_optimizer(base_dir: Path, simulation_name: str, overwrite: bool) -> NSGAII_Optimizer:
    optimization_params = {
        "ParameterNames": ["x1", "x2"],
        "UpperBounds": np.array([1.0, 1.0]),
        "LowerBounds": np.array([0.0, 0.0]),
        "start_point": np.array([0.5, 0.5]),
        "n_iterations": 1,
        "n_objectives": 2,
    }
    return NSGAII_Optimizer(
        optimization_params=optimization_params,
        BaseDirectory=str(base_dir),
        SimulationName=simulation_name,
        OptimizationDirectory=base_dir,
        TopasLocation="testing_mode",
        Overwrite=overwrite,
        pop_size=4,
    )


def test_setup_does_not_prompt_for_empty_simulation_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_minimal_project(tmp_path)
    optimizer = _make_optimizer(tmp_path, "EmptyDirRun", overwrite=False)

    def _fail_input() -> str:
        raise AssertionError("input() should not be called for an empty simulation directory")

    monkeypatch.setattr("builtins.input", _fail_input)
    optimizer.SetUpDirectoryStructure()
    assert (tmp_path / "EmptyDirRun" / "logs").exists()


def test_nonempty_directory_without_overwrite_raises_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_minimal_project(tmp_path)
    sim_dir = tmp_path / "NonemptyDirRun"
    sim_dir.mkdir()
    (sim_dir / "stale.txt").write_text("stale")

    optimizer = _make_optimizer(tmp_path, "NonemptyDirRun", overwrite=False)

    def _fail_input() -> str:
        raise AssertionError(
            "input() must not be called from the library; the non-empty "
            "directory path should raise RuntimeError unconditionally"
        )

    monkeypatch.setattr("builtins.input", _fail_input)
    with pytest.raises(RuntimeError):
        optimizer.SetUpDirectoryStructure()


def test_evaluate_objectives_rejects_missing_dict_parameters(tmp_path: Path) -> None:
    _write_minimal_project(tmp_path)
    optimizer = _make_optimizer(tmp_path, "MissingDictInput", overwrite=True)
    optimizer.SetUpDirectoryStructure()

    with pytest.raises(InvalidParameterError):
        optimizer.EvaluateObjectives({"x1": 0.1})


def test_evaluate_objectives_rejects_non_1d_objective_return(tmp_path: Path) -> None:
    _write_minimal_project(tmp_path)
    optimizer = _make_optimizer(tmp_path, "BadObjectiveShape", overwrite=True)
    optimizer.SetUpDirectoryStructure()
    optimizer.TopasObjectiveFunction = lambda *_args, **_kwargs: [[0.1, 0.2]]

    with pytest.raises(ObjectiveFunctionError):
        optimizer.EvaluateObjectives(np.array([0.2, 0.8]))


def test_default_optimizer_style_split(tmp_path: Path) -> None:
    _write_minimal_project(tmp_path)
    optimizer = _make_optimizer(tmp_path, "DefaultStyles", overwrite=True)
    assert optimizer.intermediate_plot_style == "fast"
    assert optimizer.plot_style == "publication"
    assert optimizer.publication_variant == "clean"


def test_invalid_optimizer_style_raises(tmp_path: Path) -> None:
    _write_minimal_project(tmp_path)
    optimization_params = {
        "ParameterNames": ["x1", "x2"],
        "UpperBounds": np.array([1.0, 1.0]),
        "LowerBounds": np.array([0.0, 0.0]),
        "start_point": np.array([0.5, 0.5]),
        "n_iterations": 1,
        "n_objectives": 2,
    }
    with pytest.raises(InvalidParameterError):
        NSGAII_Optimizer(
            optimization_params=optimization_params,
            BaseDirectory=str(tmp_path),
            SimulationName="BadStyle",
            OptimizationDirectory=tmp_path,
            TopasLocation="testing_mode",
            Overwrite=True,
            plot_style="journal",
        )


def test_invalid_publication_variant_raises(tmp_path: Path) -> None:
    _write_minimal_project(tmp_path)
    optimization_params = {
        "ParameterNames": ["x1", "x2"],
        "UpperBounds": np.array([1.0, 1.0]),
        "LowerBounds": np.array([0.0, 0.0]),
        "start_point": np.array([0.5, 0.5]),
        "n_iterations": 1,
        "n_objectives": 2,
    }
    with pytest.raises(InvalidParameterError):
        NSGAII_Optimizer(
            optimization_params=optimization_params,
            BaseDirectory=str(tmp_path),
            SimulationName="BadVariant",
            OptimizationDirectory=tmp_path,
            TopasLocation="testing_mode",
            Overwrite=True,
            publication_variant="journal",
        )


@pytest.mark.parametrize("variant", ["clean", "nature", "ieee", "medicalphysics"])
def test_publication_variants_accepted(tmp_path: Path, variant: str) -> None:
    _write_minimal_project(tmp_path)
    optimization_params = {
        "ParameterNames": ["x1", "x2"],
        "UpperBounds": np.array([1.0, 1.0]),
        "LowerBounds": np.array([0.0, 0.0]),
        "start_point": np.array([0.5, 0.5]),
        "n_iterations": 1,
        "n_objectives": 2,
    }
    optimizer = NSGAII_Optimizer(
        optimization_params=optimization_params,
        BaseDirectory=str(tmp_path),
        SimulationName=f"Variant_{variant}",
        OptimizationDirectory=tmp_path,
        TopasLocation="testing_mode",
        Overwrite=True,
        publication_variant=variant,
    )
    assert optimizer.publication_variant == variant


def test_verbose_default_is_false(tmp_path: Path) -> None:
    _write_minimal_project(tmp_path)
    optimizer = _make_optimizer(tmp_path, "VerboseDefault", overwrite=True)
    assert optimizer.verbose is False


def test_verbose_can_be_enabled(tmp_path: Path) -> None:
    _write_minimal_project(tmp_path)
    optimization_params = {
        "ParameterNames": ["x1", "x2"],
        "UpperBounds": np.array([1.0, 1.0]),
        "LowerBounds": np.array([0.0, 0.0]),
        "start_point": np.array([0.5, 0.5]),
        "n_iterations": 1,
        "n_objectives": 2,
    }
    optimizer = NSGAII_Optimizer(
        optimization_params=optimization_params,
        BaseDirectory=str(tmp_path),
        SimulationName="VerboseExplicit",
        OptimizationDirectory=tmp_path,
        TopasLocation="testing_mode",
        Overwrite=True,
        verbose=True,
    )
    assert optimizer.verbose is True


def test_missing_objective_function_raises_clear_error(tmp_path: Path) -> None:
    """A user project missing the required callable fails up front, not later."""
    (tmp_path / "GenerateTopasScripts.py").write_text(
        "def GenerateTopasScripts(BaseDirectory, iteration, **variable_dict):\n"
        "    return [['# script']], ['Sim']\n"
    )
    # Defines the module but not the required TopasObjectiveFunction callable.
    (tmp_path / "TopasObjectiveFunction.py").write_text("WRONG_NAME = 1\n")

    optimization_params = {
        "ParameterNames": ["x1", "x2"],
        "UpperBounds": np.array([1.0, 1.0]),
        "LowerBounds": np.array([0.0, 0.0]),
        "start_point": np.array([0.5, 0.5]),
        "n_iterations": 1,
        "n_objectives": 2,
    }
    with pytest.raises(InvalidParameterError, match="TopasObjectiveFunction"):
        NSGAII_Optimizer(
            optimization_params=optimization_params,
            BaseDirectory=str(tmp_path),
            SimulationName="MissingObjective",
            OptimizationDirectory=tmp_path,
            TopasLocation="testing_mode",
            Overwrite=True,
        )


def test_no_stream_handler_attached_at_import() -> None:
    """The library must not attach a StreamHandler at import time.

    Doing so would hijack the root logger's output and make the package
    unfriendly to embed-style users.
    """
    import logging

    from TopasMOO import optimizers as opt_mod

    pkg_logger = logging.getLogger(opt_mod.__name__)
    assert pkg_logger.handlers == [], (
        "TopasMOO.optimizers attached %r at import time" % pkg_logger.handlers
    )
