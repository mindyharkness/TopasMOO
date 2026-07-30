"""Smoke tests for MOBOOptimizer (BoTorch optional extra).

These are intentionally small (2-variable toy, 2 acquisition batches) so they
stay suitable for CI. Slow ZDT1/BNH campaigns live under ``benchmarks/``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("botorch")
pytest.importorskip("torch")

from TopasMOO.mobo import MOBOOptimizer, _tensor_to_float
from TopasMOO.plotting.comprehensive import RunData

# temp_dir / opt_dir come from tests/conftest.py.


def _toy_objective(X: np.ndarray) -> np.ndarray:
    """Simple bi-objective: minimize (x0^2 + x1^2, (x0-1)^2 + (x1-1)^2)."""
    X = np.atleast_2d(X)
    f1 = np.sum(X**2, axis=1)
    f2 = np.sum((X - 1.0) ** 2, axis=1)
    return np.column_stack([f1, f2])


def _make_mobo(temp_dir, opt_dir, **kwargs):
    params = {
        "ParameterNames": ["x1", "x2"],
        "UpperBounds": np.array([1.0, 1.0]),
        "LowerBounds": np.array([0.0, 0.0]),
        "start_point": np.array([0.5, 0.5]),
        "n_generations": kwargs.pop("n_batches", 2),
        "n_objectives": 2,
    }
    defaults = dict(
        optimization_params=params,
        BaseDirectory=temp_dir,
        SimulationName="mobo_smoke",
        OptimizationDirectory=opt_dir,
        TopasLocation="testing_mode",
        Overwrite=True,
        KeepAllResults=False,
        n_init=6,
        batch_size=2,
        seed=0,
        num_restarts=2,
        raw_samples=32,
        objective_fn=_toy_objective,
        acquisition="qlognehvi",
        final_plots=None,
        plot_frequency=10_000,
    )
    defaults.update(kwargs)
    return MOBOOptimizer(**defaults)


class TestTensorHelper:
    def test_tensor_to_float_roundtrip(self):
        import torch

        t = torch.tensor(3.5, dtype=torch.double)
        assert _tensor_to_float(t) == 3.5
        assert isinstance(_tensor_to_float(t), float)


class TestSignRoundTrip:
    def test_sign_flip_round_trip(self):
        Y = np.array([[1.0, 2.0], [0.5, 3.0]])
        assert np.allclose(
            MOBOOptimizer._from_botorch_objectives(
                MOBOOptimizer._to_botorch_objectives(Y)
            ),
            Y,
        )


class TestMOBOSmoke:
    def test_loop_checkpoint_and_reload(self, temp_dir, opt_dir):
        opt = _make_mobo(temp_dir, opt_dir, SimulationName="mobo_smoke_a")
        opt.SetUpDirectoryStructure()
        result = opt.run(n_batches=2)

        assert result.F.ndim == 2
        assert result.X.ndim == 2
        assert len(opt.HypervolumeHistory) >= 1
        ckpt = Path(opt._mobo_ckpt_path())
        assert ckpt.is_file()

        opt2 = _make_mobo(
            temp_dir,
            opt_dir,
            SimulationName="mobo_smoke_a",
            Overwrite=False,
            resume=True,
            n_batches=2,
        )
        assert opt2.load_checkpoint()
        assert opt2.train_X is not None
        assert len(opt2.train_X) == len(opt.train_X)
        assert np.allclose(opt2.train_Y, opt.train_Y)

    def test_results_feed_rundata(self, temp_dir, opt_dir):
        opt = _make_mobo(temp_dir, opt_dir, SimulationName="mobo_smoke_plot")
        opt.SetUpDirectoryStructure()
        opt.run(n_batches=1)
        data = RunData.from_optimizer(opt)
        assert data.pareto_objectives.shape[1] == 2
        assert len(data.hypervolume_history) >= 1
        assert data.n_objectives == 2

    def test_seed_reproducibility(self, temp_dir, opt_dir):
        a = _make_mobo(temp_dir, opt_dir, SimulationName="mobo_seed_a", seed=123)
        a.SetUpDirectoryStructure()
        Xa = a.ask()

        b = _make_mobo(temp_dir, opt_dir, SimulationName="mobo_seed_b", seed=123)
        b.SetUpDirectoryStructure()
        Xb = b.ask()
        assert np.allclose(Xa, Xb)

    def test_start_point_injected_like_nsga(self, temp_dir, opt_dir):
        opt = _make_mobo(temp_dir, opt_dir, SimulationName="mobo_start", n_init=6)
        opt.SetUpDirectoryStructure()
        X0 = opt.ask()
        assert np.allclose(X0[0], opt.StartingValues)

    def test_pareto_front_files_match_nsga_contract(self, temp_dir, opt_dir):
        """Running + final Pareto files use the same split as NSGA-II."""
        opt = _make_mobo(temp_dir, opt_dir, SimulationName="mobo_pareto_files")
        opt.SetUpDirectoryStructure()
        result = opt.run(n_batches=1)
        running = Path(opt._ParetoRunningLogFileLoc)
        final = Path(opt._ParetoLogFileLoc)
        assert running.is_file()
        assert final.is_file()
        assert result.F.shape == np.asarray(opt.ParetoObjectives).shape
        assert np.allclose(result.F, opt.ParetoObjectives)
        assert np.allclose(result.X, opt.ParetoDecisionVars)

    def test_n_generations_means_acquisition_batches(self, temp_dir, opt_dir):
        opt = _make_mobo(temp_dir, opt_dir, SimulationName="mobo_batches", n_batches=3)
        assert opt.n_generations == 3
        assert opt._n_batches_target == 3
