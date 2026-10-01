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

from TopasMOO.exceptions import InvalidParameterError
from TopasMOO.mobo import MOBOOptimizer
from TopasMOO.plotting.comprehensive import RunData

# temp_dir / opt_dir come from tests/conftest.py.


def _toy_objective(X: np.ndarray) -> np.ndarray:
    """Simple bi-objective: minimize (x0^2 + x1^2, (x0-1)^2 + (x1-1)^2)."""
    X = np.atleast_2d(X)
    f1 = np.sum(X**2, axis=1)
    f2 = np.sum((X - 1.0) ** 2, axis=1)
    return np.column_stack([f1, f2])


def _constrained_objective(X):
    return np.column_stack([_toy_objective(X), X[:, 0] - 0.6, 0.1 - X[:, 1]])


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


class TestSignRoundTrip:
    def test_topasmoo_minimization_is_negated_for_botorch(self):
        Y = np.array([[1.0, 2.0], [0.5, 3.0]])
        Y_botorch = MOBOOptimizer._to_botorch_objectives(Y)

        np.testing.assert_array_equal(Y_botorch, -Y)
        assert np.argmin(Y[:, 0]) == np.argmax(Y_botorch[:, 0])
        assert np.allclose(
            MOBOOptimizer._from_botorch_objectives(Y_botorch),
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
        assert opt.gp_prediction_history is not None
        assert opt2.gp_prediction_history is not None
        assert opt.gp_prediction_history.shape == opt.train_Y.shape
        assert np.all(np.isnan(opt.gp_prediction_history[: opt.n_init]))
        assert np.all(np.isfinite(opt.gp_prediction_history[opt.n_init :]))
        assert np.allclose(
            opt2.gp_prediction_history,
            opt.gp_prediction_history,
            equal_nan=True,
        )

        data = RunData.from_optimizer(opt)
        assert data.pareto_objectives.shape[1] == 2
        assert len(data.hypervolume_history) >= 1
        assert data.n_objectives == 2
        assert data.observed_objectives is opt.train_Y
        assert data.gp_prediction_history is opt.gp_prediction_history
        assert data.failed_mask is opt.train_failed

        running = Path(opt._ParetoRunningLogFileLoc)
        final = Path(opt._ParetoLogFileLoc)
        assert running.is_file()
        assert final.is_file()
        assert result.F.shape == np.asarray(opt.ParetoObjectives).shape
        assert np.allclose(result.F, opt.ParetoObjectives)
        assert np.allclose(result.X, opt.ParetoDecisionVars)

    def test_posterior_predictions_return_to_minimization_space(self, temp_dir, opt_dir):
        import torch

        class _Posterior:
            mean = torch.tensor([[-2.0, -3.0]], dtype=torch.double)

        class _Model:
            def eval(self):
                return self

            def posterior(self, _X):
                return _Posterior()

        opt = _make_mobo(temp_dir, opt_dir, SimulationName="mobo_prediction_sign")
        prediction = opt._posterior_mean_minimization(
            np.array([[0.25, 0.75]]),
            model=_Model(),
        )
        np.testing.assert_array_equal(prediction, [[2.0, 3.0]])

    def test_old_checkpoint_without_prediction_history_loads(self, temp_dir, opt_dir):
        opt = _make_mobo(temp_dir, opt_dir, SimulationName="mobo_old_prediction_ckpt")
        opt.SetUpDirectoryStructure()
        opt.run(n_batches=1)
        checkpoint = Path(opt._mobo_ckpt_path())

        with np.load(checkpoint, allow_pickle=False) as stored:
            legacy = {
                key: np.asarray(stored[key]).copy()
                for key in stored.files
                if key not in {"gp_prediction_history", "pending_gp_predictions"}
            }
        np.savez_compressed(checkpoint, **legacy)

        restored = _make_mobo(
            temp_dir,
            opt_dir,
            SimulationName="mobo_old_prediction_ckpt",
            Overwrite=False,
            resume=True,
        )
        assert restored.load_checkpoint()
        assert restored.gp_prediction_history is not None
        assert restored.gp_prediction_history.shape == restored.train_Y.shape
        assert np.all(np.isnan(restored.gp_prediction_history))

    def test_seed_reproducibility(self, temp_dir, opt_dir):
        a = _make_mobo(temp_dir, opt_dir, SimulationName="mobo_seed_a", seed=123)
        a.SetUpDirectoryStructure()
        Xa = a.ask()

        b = _make_mobo(temp_dir, opt_dir, SimulationName="mobo_seed_b", seed=123)
        b.SetUpDirectoryStructure()
        Xb = b.ask()
        assert np.allclose(Xa, Xb)

    def test_pending_initial_predictions_survive_checkpoint(self, temp_dir, opt_dir):
        opt = _make_mobo(temp_dir, opt_dir, SimulationName="mobo_pending_predictions")
        X = opt.ask()
        assert opt._pending_gp_predictions is not None
        assert np.all(np.isnan(opt._pending_gp_predictions))

        restored = _make_mobo(
            temp_dir,
            opt_dir,
            SimulationName="mobo_pending_predictions",
            Overwrite=False,
            resume=True,
        )
        assert restored.load_checkpoint()
        assert np.array_equal(restored._pending_X, X)
        assert restored._pending_gp_predictions is not None
        assert restored._pending_gp_predictions.shape == (opt.n_init, opt.n_objectives)
        assert np.all(np.isnan(restored._pending_gp_predictions))

    def test_start_point_injected_like_nsga(self, temp_dir, opt_dir):
        opt = _make_mobo(temp_dir, opt_dir, SimulationName="mobo_start", n_init=6)
        opt.SetUpDirectoryStructure()
        X0 = opt.ask()
        assert np.allclose(X0[0], opt.StartingValues)

    def test_reported_front_minimizes_like_nsga(self, temp_dir, opt_dir):
        opt = _make_mobo(temp_dir, opt_dir, SimulationName="mobo_minimizes")
        opt.SetUpDirectoryStructure()

        X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        Y = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        opt.tell(X, Y)

        # pymoo/NSGA-II also treats [1, 1] as dominating the larger rows.
        np.testing.assert_array_equal(opt.ParetoObjectives, [[1.0, 1.0]])
        np.testing.assert_array_equal(opt.ParetoDecisionVars, [[0.0, 0.0]])

    def test_n_generations_means_acquisition_batches(self, temp_dir, opt_dir):
        opt = _make_mobo(temp_dir, opt_dir, SimulationName="mobo_batches", n_batches=3)
        assert opt.n_generations == 3
        assert opt._n_batches_target == 3


@pytest.mark.parametrize("acquisition", ["qlognehvi", "qlognparego"])
@pytest.mark.parametrize("initial_feasible", [True, False])
def test_measured_constraints_acquisition_and_resume(
    temp_dir, opt_dir, monkeypatch, acquisition, initial_feasible
):
    import torch

    import TopasMOO.mobo as mobo

    settings = dict(
        n_constraints=2,
        acquisition=acquisition,
        objective_fn=_constrained_objective,
        decision_constraints=[lambda x: x.sum() - 1.0] if initial_feasible else [],
    )
    opt = _make_mobo(temp_dir, opt_dir, **settings)
    opt.SetUpDirectoryStructure()
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.75, 0.05], [0.8, 0.1]])
    if not initial_feasible:
        X = np.array([[0.65, 0.05], [0.7, 0.07], [0.8, 0.09]])
    opt.tell(X, _constrained_objective(X))
    np.testing.assert_array_equal(
        opt.feasible_mask(), [True, True, False, False] if initial_feasible else [False] * 3
    )
    if not initial_feasible:
        assert opt.ParetoObjectives.shape == (0, 2)
        assert opt.HypervolumeHistory == [0.0]
        assert opt._hv_ref_fixed is None

    original_build = mobo.build_acquisition
    acquisitions = []

    def checked_build(*args, **kwargs):
        acq = original_build(*args, **kwargs)
        samples = torch.tensor([[[-2.0, -3.0, -0.2, 0.3]]], dtype=torch.double)
        constraints = acq._constraints if acquisition == "qlognparego" else acq.constraints
        assert [float(g(samples).item()) for g in constraints] == [-0.2, 0.3]
        acquisitions.append(acq)
        return acq

    monkeypatch.setattr(mobo, "build_acquisition", checked_build)
    result = opt.run(n_batches=1)
    assert acquisitions
    assert opt.model.num_outputs == 4
    # Verify actual GP training targets: only the two objectives are negated.
    expected = np.column_stack([-_toy_objective(X), _constrained_objective(X)[:, 2:]])
    for j, gp in enumerate(opt.model.models):
        raw, _ = gp.outcome_transform.untransform(gp.train_targets.unsqueeze(-1))
        np.testing.assert_allclose(raw.detach().numpy().ravel(), expected[:, j], atol=1e-10)
    assert opt.gp_prediction_history.shape == opt.train_Y.shape
    assert np.isfinite(opt.gp_prediction_history[len(X) :]).all()
    if initial_feasible:
        assert np.all(opt.train_X[len(X) :].sum(axis=1) <= 1.0 + 1e-6)
    assert result.F.shape[1] == 2
    assert np.all(_constrained_objective(result.X)[:, 2:] <= 0)
    np.testing.assert_array_equal(opt._gp_training_rows(), np.arange(len(opt.train_Y)))

    restored = _make_mobo(temp_dir, opt_dir, Overwrite=False, resume=True, **settings)
    assert restored.load_checkpoint()
    np.testing.assert_array_equal(restored.train_G, opt.train_G)
    np.testing.assert_array_equal(restored.feasible_mask(), opt.feasible_mask())
    np.testing.assert_allclose(restored.ParetoObjectives, opt.ParetoObjectives)
    np.testing.assert_allclose(restored.HypervolumeHistory, opt.HypervolumeHistory)


def test_constraint_validation_is_atomic_and_variances_are_objective_only(temp_dir, opt_dir):
    opt = _make_mobo(temp_dir, opt_dir, n_constraints=2)
    opt.SetUpDirectoryStructure()
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    Y = _constrained_objective(X)
    Y[0, 2:] = 0.0  # The boundary is feasible, matching pymoo's g <= 0 contract.
    V = np.full((2, 2), 0.01)
    opt.tell(X, Y, Yvar=V)
    assert opt.feasible_mask().all()
    for bad in [Y[:, :2], np.full((2, 4), np.nan), Y[:, None, :]]:
        with pytest.raises(InvalidParameterError):
            opt.tell(X, bad, Yvar=np.full((2, 2), -1.0))
        np.testing.assert_array_equal(opt.train_G, Y[:, 2:])
        np.testing.assert_array_equal(opt.train_Yvar, V)
        assert not opt._mc_uncertainty_fallback
        assert opt._batch_index == 1
    with pytest.raises(InvalidParameterError, match="objective shape"):
        opt.tell(X, Y, Yvar=np.ones((2, 4)))
    opt.tell(X, Y, Yvar=V)
    model = opt._fit_model()
    assert model.num_outputs == 4
    assert opt.train_Yvar.shape == (4, 2)


def test_topas_constraints_cache_failures_and_running_front(temp_dir, opt_dir, monkeypatch):
    opt = _make_mobo(temp_dir, opt_dir, n_constraints=2, objective_fn=None, plot_frequency=1)
    opt.SetUpDirectoryStructure()
    monkeypatch.setattr(opt, "_plot_convergence", lambda: None)
    calls = []

    def evaluate(_path, _iteration):
        calls.append(1)
        if opt.x[0, 0] > 0.9:
            raise RuntimeError("simulation output unavailable")
        return _constrained_objective(opt.x)[0].tolist()

    opt.TopasObjectiveFunction = evaluate
    X = np.array([[0.2, 0.2], [0.75, 0.2], [0.95, 0.2]])
    Y, V, failed = opt._evaluate_batch(X)
    np.testing.assert_array_equal(failed, [False, False, True])
    assert len(calls) == 3
    # The batch has not reached tell(): monitoring must already exclude the
    # violated constraint and the failure, using the raw evaluation cache.
    np.testing.assert_array_equal(opt.ParetoDecisionVars, X[:1])
    np.testing.assert_array_equal(opt._eligible_rows(3), [True, False, False])
    opt.tell(X, Y, V, failed=failed)
    np.testing.assert_array_equal(opt._gp_training_rows(), [0, 1])
    np.testing.assert_array_equal(opt.train_G[:2], Y[:2, 2:])

    restored = _make_mobo(
        temp_dir, opt_dir, n_constraints=2, objective_fn=None, Overwrite=False, resume=True
    )
    restored.SetUpDirectoryStructure()
    restored.TopasObjectiveFunction = lambda *_: pytest.fail("cached evaluation was rerun")
    cached, _, cached_failed = restored._evaluate_batch(X)
    np.testing.assert_array_equal(cached, Y)
    np.testing.assert_array_equal(cached_failed, failed)
    np.testing.assert_array_equal(restored._eligible_rows(6), [True, False, False] * 2)


def test_running_front_checks_each_inflight_design_once(temp_dir, opt_dir, monkeypatch):
    calls = []

    def constraint(x):
        calls.append(1)
        return x[0] - 2.0

    opt = _make_mobo(
        temp_dir,
        opt_dir,
        n_constraints=2,
        objective_fn=None,
        plot_frequency=1,
        decision_constraints=[constraint],
    )
    opt.SetUpDirectoryStructure()
    monkeypatch.setattr(opt, "_plot_convergence", lambda: None)
    opt.TopasObjectiveFunction = lambda *_: _constrained_objective(opt.x)[0].tolist()
    X = np.column_stack([np.linspace(0.1, 0.5, 6), np.full(6, 0.2)])
    opt._evaluate_batch(X)
    # Every evaluation refreshes the running front; without the per-design
    # cache this grew quadratically (1 + 2 + ... + 6 = 21 calls).
    assert len(calls) == len(X)


@pytest.mark.parametrize("corruption", ["missing", "shape", "nonfinite"])
def test_constraint_checkpoint_rejected_before_restore(temp_dir, opt_dir, corruption):
    opt = _make_mobo(temp_dir, opt_dir, n_constraints=2)
    opt.SetUpDirectoryStructure()
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    opt.tell(X, _constrained_objective(X))
    path = opt._mobo_ckpt_path()
    with np.load(path) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    if corruption == "missing":
        del arrays["train_G"]
    elif corruption == "shape":
        arrays["train_G"] = np.zeros((1, 2))
    else:
        arrays["train_G"][0, 0] = np.nan
    np.savez_compressed(path, **arrays)
    restored = _make_mobo(temp_dir, opt_dir, n_constraints=2, Overwrite=False, resume=True)
    with pytest.raises(InvalidParameterError, match="constraint|train_G"):
        restored.load_checkpoint()
    assert restored.train_X is None


def test_legacy_unconstrained_checkpoint_stays_unconstrained(temp_dir, opt_dir):
    opt = _make_mobo(temp_dir, opt_dir)
    opt.SetUpDirectoryStructure()
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    opt.tell(X, _toy_objective(X))
    path = opt._mobo_ckpt_path()
    with np.load(path) as stored:
        arrays = {key: stored[key].copy() for key in stored.files if key != "train_G"}
    np.savez_compressed(path, **arrays)
    restored = _make_mobo(temp_dir, opt_dir, Overwrite=False, resume=True)
    assert restored.load_checkpoint()
    assert restored.train_G.shape == (2, 0)
    constrained = _make_mobo(temp_dir, opt_dir, n_constraints=2, Overwrite=False, resume=True)
    with pytest.raises(InvalidParameterError, match="train_G"):
        constrained.load_checkpoint()


@pytest.mark.parametrize("recovers", [False, True])
def test_failed_initial_sampling_uses_batch_budget(temp_dir, opt_dir, recovers):
    opt = _make_mobo(temp_dir, opt_dir, n_constraints=2, objective_fn=None)
    opt.SetUpDirectoryStructure()

    def fail(*_):
        if not recovers or opt.evaluation_index < opt.n_init:
            raise RuntimeError("no usable output")
        return _constrained_objective(opt.x)[0].tolist()

    opt.TopasObjectiveFunction = fail
    if recovers:
        opt.run(n_batches=2)
        assert opt.model.num_outputs == 4
        assert opt.train_failed.sum() == opt.n_init
        np.testing.assert_array_equal(
            opt._gp_training_rows(), np.arange(opt.n_init, len(opt.train_X))
        )
    else:
        with pytest.raises(RuntimeError, match="budget exhausted"):
            opt.run(n_batches=2)
        assert len(opt._gp_training_rows()) == 0
        assert opt.HypervolumeHistory == [0.0, 0.0, 0.0]
        assert opt.ParetoObjectives.shape == (0, 2)
    assert len(opt.train_X) == opt.n_init + 2 * opt.batch_size
    initial_and_recovery = opt.train_X[: opt.n_init + opt.batch_size]
    assert len(np.unique(initial_and_recovery, axis=0)) == len(initial_and_recovery)
