# -*- coding: utf-8 -*-
"""
Multi-objective Bayesian optimization (MOBO) for TopasMOO via BoTorch.

Provides ``MOBOOptimizer``, an alternative to ``NSGAII_Optimizer`` for
when you don't want to run as many TOPAS simulations (roughly 100--500 evaluations, parameter count
comfortably below ~15). Bayesian optimization more efficiently explores the parameter space than NSGA-II.

Uses a shared Gaussian-process backend with
``qLogNEHVI`` and ``qLogNParEGO`` acquisitions.

Shares the base-class conventions with the pymoo drivers: objectives are
minimized, inequality constraints use ``g(x) <= 0``, and
``optimization_params['n_generations']`` counts algorithm steps (here:
acquisition batches after the initial design).

Requires the optional ``mobo`` extra (so users who just use the pymoo implementation
don't need to install BoTorch + its dependencies):

    uv sync --extra mobo

BoTorch maximizes objectives; TopasMOO minimizes. The sign flip happens in
exactly one place (`MOBOOptimizer._to_botorch_objectives`) and is undone
before results are handed to plotting.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal, Sequence

import numpy as np
from pymoo.indicators.hv import HV

from .exceptions import InvalidParameterError
from .io import LogParetoFrontToFile
from .metrics import hypervolume_reference_point
from .optimizers import TopasMOOBaseClass
from .utilities import _tensor_to_float

logger = logging.getLogger(__name__)

# load acquisitions from botorch
AcquisitionName = Literal["auto", "qlognehvi", "qlognparego"]

#: Sentinel distinguishing "not looked up yet" from a resolved absence (None).
_UNRESOLVED = object()

def _require_botorch():
    """
    Import botorch stack cleanly, raise a clear optional-extra error.
    """
    try:
        import torch
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement,
        )
        from botorch.fit import fit_gpytorch_mll
        from botorch.models.gp_regression import SingleTaskGP
        from botorch.models.model_list_gp_regression import ModelListGP
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim import optimize_acqf
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.utils.multi_objective.hypervolume import infer_reference_point
        from botorch.utils.multi_objective.pareto import is_non_dominated
        from botorch.utils.sampling import draw_sobol_samples
        from gpytorch.mlls import SumMarginalLogLikelihood
    except ImportError as exc:
        raise ImportError(
            "MOBOOptimizer requires the optional 'mobo' extra. "
            "Install it with `uv sync --extra mobo` "
            "(or `pip install TopasMOO[mobo]`)."
        ) from exc

    # qLogNParEGO may live in slightly different modules across BoTorch versions.
    qLogNParEGO = None
    try:
        from botorch.acquisition.multi_objective.parego import qLogNParEGO as _parego

        qLogNParEGO = _parego
    except ImportError:
        try:
            from botorch.acquisition.multi_objective import qLogNParEGO as _parego

            qLogNParEGO = _parego
        except ImportError:
            qLogNParEGO = None

    return {
        "torch": torch,
        "qLogNoisyExpectedHypervolumeImprovement": qLogNoisyExpectedHypervolumeImprovement,
        "qLogNParEGO": qLogNParEGO,
        "fit_gpytorch_mll": fit_gpytorch_mll,
        "SingleTaskGP": SingleTaskGP,
        "ModelListGP": ModelListGP,
        "Normalize": Normalize,
        "Standardize": Standardize,
        "optimize_acqf": optimize_acqf,
        "SobolQMCNormalSampler": SobolQMCNormalSampler,
        "infer_reference_point": infer_reference_point,
        "is_non_dominated": is_non_dominated,
        "draw_sobol_samples": draw_sobol_samples,
        "SumMarginalLogLikelihood": SumMarginalLogLikelihood,
    }

def _resolve_acquisition(
    acquisition: AcquisitionName,
    n_objectives: int,
) -> str:
    """Return concrete acquisition key and log auto-selection reason."""
    key = acquisition.lower().strip()
    if key == "auto":
        if n_objectives <= 4:
            chosen = "qlognehvi"
            reason = (
                f"n_objectives={n_objectives} is in [2, 4]; "
                "selecting qLogNEHVI for hypervolume improvement."
            )
        else:
            chosen = "qlognparego"
            reason = f"n_objectives={n_objectives} is >= 5; selecting qLogNParEGO for scalability."
        logger.info("acquisition='auto' → %s (%s)", chosen, reason)
        return chosen
    if key in {"qlognehvi", "qlognparego"}:
        return key
    raise InvalidParameterError(
        f"acquisition must be 'auto', 'qlognehvi', or 'qlognparego'. Got {acquisition!r}."
    )


def build_acquisition(
    name: str,
    model,
    train_X,
    *,
    ref_point,
    sampler,
    scalarization_weights=None,
    X_pending=None,
    bt=None,
):
    """Build multi-objective acquisition objects.

    GP fitting, ``optimize_acqf``, checkpointing, and logging are intentionally
    outside this function, so acquisition choice does not touch that code.

    Constraints are not applied here: TopasMOO enforces feasibility with
    ``decision_constraints`` during ``optimize_acqf``, so the model's outputs
    are exactly the objectives and BoTorch's default identity objective applies.

    :param name: ``"qlognehvi"`` or ``"qlognparego"``.
    :param model: Fitted ``ModelListGP``.
    :param train_X: Observed inputs ``(n, d)``.
    :param ref_point: Reference point in BoTorch (maximize) space (NEHVI only).
    :param sampler: MC sampler.
    :param scalarization_weights: Optional ``m``-vector for ParEGO; if omitted,
        BoTorch samples once from the unit simplex. Callers that need
        per-candidate redraws should pass a fresh weight vector each time.
    :param X_pending: Pending points (ParEGO sequential batches).
    :param bt: Already-imported BoTorch symbol table from
        :func:`_require_botorch`; imported on demand when omitted.
    """
    if bt is None:
        bt = _require_botorch()
    key = name.lower()

    if key == "qlognehvi":
        kwargs: dict[str, Any] = {
            "model": model,
            "ref_point": ref_point,
            "X_baseline": train_X,
            "sampler": sampler,
            "prune_baseline": True,
        }
        return bt["qLogNoisyExpectedHypervolumeImprovement"](**kwargs)

    if key == "qlognparego":
        cls = bt["qLogNParEGO"]
        if cls is None:
            raise ImportError(
                "qLogNParEGO is not available in this BoTorch install. "
                "Upgrade botorch or select acquisition='qlognehvi'."
            )
        kwargs = {
            "model": model,
            "X_baseline": train_X,
            "sampler": sampler,
            "prune_baseline": True,
        }
        if scalarization_weights is not None:
            kwargs["scalarization_weights"] = scalarization_weights
        if X_pending is not None:
            kwargs["X_pending"] = X_pending
        return cls(**kwargs)

    # we only support these two acq functions
    raise InvalidParameterError(f"Unknown acquisition {name!r}.")


def propagate_objective_variance(
    values: Sequence[float],
    raw_variances: Sequence[float],
    *,
    jacobian: Sequence[Sequence[float]] | np.ndarray | None = None,
    covariance: Sequence[Sequence[float]] | np.ndarray | None = None,
    independence: bool = True,
) -> np.ndarray:
    """Propagate scorer uncertainty through an objective transformation.

    For raw scored quantities ``z`` and objectives ``f(z)``, first-order
    propagation is ``Cov[f] ≈ J Cov[z] Jᵀ``, where ``J`` is the Jacobian of
    ``f`` evaluated at ``values``. With no ``jacobian``, the identity mapping is
    assumed and the variances are returned unchanged.

    The default assumes independent scored quantities and constructs
    ``Cov[z]`` from ``raw_variances``. Set ``independence=False`` and provide a
    full covariance matrix when cross terms matter.

    :param values: Raw scored means at which ``jacobian`` was evaluated.
    :param raw_variances: Per-scorer variances.
    :param jacobian: Optional array with shape ``(n_objectives, n_scorers)``.
    :param covariance: Full scorer covariance, required when
        ``independence=False``.
    :param independence: Whether scorer covariance is diagonal.
    :returns: Propagated objective variances as a 1-D float array.
    """
    means = np.asarray(values, dtype=float).reshape(-1)
    var = np.asarray(raw_variances, dtype=float).reshape(-1)
    if means.shape != var.shape:
        raise ValueError(
            f"values and raw_variances must have the same shape. Got {means.shape} and {var.shape}."
        )
    if not np.all(np.isfinite(means)):
        raise ValueError(f"values must be finite. Got {values!r}.")
    if not np.all(np.isfinite(var)) or np.any(var < 0):
        raise ValueError(f"raw_variances must be finite and non-negative. Got {raw_variances!r}.")

    if jacobian is None:
        J = np.eye(len(means), dtype=float)
    else:
        J = np.atleast_2d(np.asarray(jacobian, dtype=float))
        if J.shape[1] != len(means) or not np.all(np.isfinite(J)):
            raise ValueError(
                f"jacobian must be finite with shape (n_objectives, {len(means)}). Got {J.shape}."
            )

    if independence:
        cov = np.diag(var)
    else:
        if covariance is None:
            raise NotImplementedError(
                "independence=False requires a full scorer covariance matrix."
            )
        cov = np.asarray(covariance, dtype=float)
        if cov.shape != (len(means), len(means)) or not np.all(np.isfinite(cov)):
            raise ValueError(
                "covariance must be finite with shape "
                f"({len(means)}, {len(means)}). Got {cov.shape}."
            )
        if not np.allclose(cov, cov.T):
            raise ValueError("covariance must be symmetric.")

    propagated = np.diag(J @ cov @ J.T)
    if np.any(propagated < -1e-12):
        raise ValueError("Propagated covariance has negative diagonal entries.")
    return np.maximum(propagated, 0.0)


class MOBOOptimizer(TopasMOOBaseClass):
    """Multi-objective Bayesian optimizer (implemented via BoTorch) for TopasMOO.

    Constructed similar to ``NSGAII_Optimizer``: shared base-class constructor
    kwargs, ``RunOptimization()`` entry point, minimization conventions, and plotting attributes.

    For this class, ``optimization_params['n_generations']`` is the number of
    acquisition batches *after* the Sobol (or rejection-sampled) initial
    design of size ``n_init``. Total evaluations are approximately
    ``n_init + n_generations * batch_size``.

    The reported Pareto front is the non-dominated set over all eligible
    observations. That differs from ``NSGAII_Optimizer``, whose front is the non-dominated set of the
    final population only (by virtue of the genetic optimization).

    Also exposes an explicit ``ask`` / ``tell`` / ``run`` API for stepwise
    evaluation (e.g. cluster batch submission)

    :param batch_size: Candidates proposed per acquisition step. Defaults to 1,
        or to ``n_parallel_jobs`` when that is provided.
    :param n_init: Sobol initial design size. Default ``max(2*d+1, 10*d)``.
    :param n_parallel_jobs: Declared concurrent TOPAS jobs; when set and
        ``batch_size`` is omitted, ``batch_size`` defaults to this value.
    :param acquisition: ``"auto"`` (default), ``"qlognehvi"``, or ``"qlognparego"``.
    :param num_restarts: Restarts for ``optimize_acqf`` (default 10).
    :param raw_samples: Raw samples for ``optimize_acqf`` (default 512).
    :param sequential: qLogNEHVI candidate optimization mode (default False).
        qLogNParEGO is always generated greedily, one candidate at a time.
        qLogNEHVI rejects ``sequential=True`` with decision constraints and
        ``batch_size > 1`` because BoTorch cannot combine that path with the
        required feasible starting points.
    :param seed: RNG seed for Sobol init, MC sampler, and acqf restarts.
    :param ref_point: Optional user reference point in **minimization** space
        (same units as ``TopasObjectiveFunction`` / ``NSGAII_Optimizer``).
    :param train_Yvar: Optional known observation noise (minimization space),
        shape ``(n_init, m)`` -- it must line up row-for-row with the first
        batch of observations, and is applied when that batch is told. A
        mismatched or non-finite array falls back to inferred noise with a
        warning. Pass ``None`` to request inferred homoskedastic noise.
    :param use_mc_uncertainty: If True, attempt to use Monte Carlo scorer
        variances as ``train_Yvar`` (opt-in research feature; default False).
    :param objective_fn: Optional callable ``(X: ndarray (n,d)) -> Y (n,m)`` for
        synthetic / benchmark loops that bypass TOPAS ``EvaluateObjectives``.
        Returned objectives must be minimized, matching the base-class contract.
    :param include_start_point: If True (default, matching ``NSGAII_Optimizer``),
        the user's ``start_point`` replaces the first row of the initial design
        so a known-good configuration is actually evaluated. Skipped with a
        warning if it violates a ``decision_constraints`` callable.
    :param decision_constraints: Optional list of callables ``g(X) -> Tensor``
        with shape matching BoTorch nonlinear inequality constraints
        (``g <= 0`` feasible on decision vectors, the TopasMOO / pymoo
        convention). Used for known analytical constraints such as BNH.
        Internally negated for BoTorch's ``callable(x) >= 0`` form.
    :param **kwds: Forwarded to ``TopasMOOBaseClass``.

    Constraints are expressed as ``decision_constraints``: analytical
    ``g(x) <= 0`` on the decision vector, enforced during acquisition
    optimization so infeasible designs are never proposed and no TOPAS run is
    spent on them. Use them to restrict which simulation parameters are
    allowed. The base class's ``n_constraints`` (measured constraint values
    modeled as extra GP outcomes) applies to ``NSGAII_Optimizer`` only, and
    ``MOBOOptimizer`` rejects it.
    """

    def __init__(
        self,
        batch_size: int | None = None,
        n_init: int | None = None,
        n_parallel_jobs: int | None = None,
        acquisition: AcquisitionName = "auto",
        num_restarts: int = 10,
        raw_samples: int = 512,
        sequential: bool = False,
        seed: int | None = None,
        ref_point: Sequence[float] | None = None,
        train_Yvar: np.ndarray | None = None,
        use_mc_uncertainty: bool = False,
        objective_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        decision_constraints: Sequence[Callable] | None = None,
        include_start_point: bool = True,
        **kwds,
    ):
        """Attach MOBO settings and delegate base initialization to the superclass.

        Every constructor parameter is documented on the class docstring;
        ``**kwds`` is forwarded to ``TopasMOOBaseClass`` (directories, params, etc.).
        """
        self.seed = seed
        self.num_restarts = int(num_restarts)
        self.raw_samples = int(raw_samples)
        self.sequential = bool(sequential)
        self.use_mc_uncertainty = bool(use_mc_uncertainty)
        self._objective_fn = objective_fn
        self._decision_constraints = list(decision_constraints or [])
        self.include_start_point = bool(include_start_point)
        self._user_ref_point_min = (
            None if ref_point is None else np.asarray(ref_point, dtype=float).reshape(-1)
        )
        self._constructor_train_Yvar = (
            None if train_Yvar is None else np.atleast_2d(np.asarray(train_Yvar, dtype=float))
        )
        # Per-design variance cache, keyed like the base-class evaluation cache,
        # so a repeated design reuses its own variances instead of re-reading
        # (possibly stale) Monte Carlo output for an iteration that never ran.
        self._variance_cache: dict[tuple[float, ...], np.ndarray] = {}
        # Memoized TopasObjectiveVariances lookup; see _objective_variance_callable.
        self._variance_fn: Any = _UNRESOLVED
        self.results_metadata: dict[str, Any] = {}

        # Base class needs directories/params before we know d for n_init default.
        super().__init__(**kwds)

        if self._objective_fn is not None and not callable(self._objective_fn):
            raise InvalidParameterError("objective_fn must be callable when supplied.")
        if self.n_constraints > 0:
            # n_constraints is a base-class feature for NSGAII_Optimizer, which
            # hands it to pymoo as n_ieq_constr. MOBO does not model measured
            # constraint values as GP outcomes; rejecting is better than
            # silently dropping constraints the caller believes are enforced.
            raise InvalidParameterError(
                f"MOBOOptimizer does not support n_constraints (got "
                f"{self.n_constraints}). Express feasibility analytically with "
                "decision_constraints=[g, ...] (g(x) <= 0 feasible), which is "
                "enforced during acquisition optimization so infeasible designs "
                "are never proposed. n_constraints remains available on "
                "NSGAII_Optimizer."
            )
        if any(not callable(g) for g in self._decision_constraints):
            raise InvalidParameterError("Every decision_constraints entry must be callable.")

        d = len(self.ParameterNames)
        if self.num_restarts < 1:
            raise InvalidParameterError(f"num_restarts must be >= 1. Got {self.num_restarts}.")
        if self.raw_samples < 1:
            raise InvalidParameterError(f"raw_samples must be >= 1. Got {self.raw_samples}.")
        if self._user_ref_point_min is not None:
            if self._user_ref_point_min.shape != (self.n_objectives,):
                raise InvalidParameterError(
                    "ref_point must contain exactly "
                    f"n_objectives={self.n_objectives} values. Got "
                    f"{self._user_ref_point_min.shape}."
                )
            if not np.all(np.isfinite(self._user_ref_point_min)):
                raise InvalidParameterError("ref_point must contain only finite values.")
        if d > 15:
            warnings.warn(
                f"d={d} parameters exceeds the comfortable regime (~15) for a standard GP!"
                "Continuing with a standard GP.",
                UserWarning,
                stacklevel=2,
            )

        if batch_size is None:
            self.batch_size = int(n_parallel_jobs) if n_parallel_jobs is not None else 1
        else:
            self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise InvalidParameterError(f"batch_size must be >= 1. Got {self.batch_size}.")
        recommended_n_init = max(2 * d + 1, 10 * d)
        if n_init is None:
            self.n_init = recommended_n_init
        else:
            self.n_init = int(n_init)
            if self.n_init < 1:
                raise InvalidParameterError(f"n_init must be >= 1. Got {self.n_init}.")
            floor = 5 * d
            if self.n_init < floor:
                warnings.warn(
                    f"n_init={self.n_init} is below the recommended floor of "
                    f"{floor} for d={d} parameters (preferred default would be "
                    f"{recommended_n_init}).",
                    UserWarning,
                    stacklevel=2,
                )

        self.acquisition = acquisition
        self._acquisition_resolved = _resolve_acquisition(acquisition, self.n_objectives)
        if (
            self._acquisition_resolved == "qlognehvi"
            and self.sequential
            and self._decision_constraints
            and self.batch_size > 1
        ):
            raise InvalidParameterError(
                "sequential=True with qLogNEHVI, decision_constraints, and "
                "batch_size > 1 is unsupported because BoTorch cannot use the "
                "required feasible batch initial conditions on its sequential "
                "path. Use sequential=False or batch_size=1."
            )

        self._bt = None  # lazy
        self._bounds_tensor = None
        self.train_X: np.ndarray | None = None  # (n, d) minimization / decision space
        self.train_Y: np.ndarray | None = None  # (n, m) TopasMOO minimize space

        self.train_failed: np.ndarray | None = None
        self.train_Yvar: np.ndarray | None = self._constructor_train_Yvar

        self._nd_front_cache: dict[int, np.ndarray] = {}

        self._feasible_cache: np.ndarray | None = None
        self._mc_uncertainty_fallback = False

        self._hv_ref_fixed: np.ndarray | None = None
        self._pending_X: np.ndarray | None = None
        self._batch_index = 0
        self._n_batches_target = int(self.n_generations)
        self._mobo_checkpoint_loc = None  # set after dirs exist
        self._meta_written = False
        self.model = None
        self.res = None

        if self.seed is not None:
            np.random.seed(self.seed)

    # Minimization-maximization sign flip (single place)

    @staticmethod
    def _to_botorch_objectives(Y_min: np.ndarray) -> np.ndarray:
        """Minimization (TopasMOO) → maximization (BoTorch)."""
        return -np.asarray(Y_min, dtype=float)

    @staticmethod
    def _to_botorch_variance(Yvar_min: np.ndarray) -> np.ndarray:
        """Variance is invariant under Y → -Y."""
        return np.asarray(Yvar_min, dtype=float)

    # BoTorch helpers

    def _ensure_botorch(self):
        if self._bt is None:
            self._bt = _require_botorch()
            torch = self._bt["torch"]
            if self.seed is not None:
                torch.manual_seed(self.seed)
            lower = np.asarray(self.LowerBounds, dtype=float).reshape(-1)
            upper = np.asarray(self.UpperBounds, dtype=float).reshape(-1)
            self._bounds_tensor = torch.tensor(np.vstack([lower, upper]), dtype=torch.double)
        return self._bt

    def _torch_X(self, X: np.ndarray):
        torch = self._ensure_botorch()["torch"]
        return torch.as_tensor(np.asarray(X, dtype=float), dtype=torch.double)

    def _torch_Y_botorch(self, Y_min: np.ndarray):
        torch = self._ensure_botorch()["torch"]
        return torch.as_tensor(self._to_botorch_objectives(Y_min), dtype=torch.double)

    def _fall_back_to_inferred_noise(self, reason: str, stacklevel: int = 3) -> None:
        """Abandon supplied observation variances for the rest of the run.

        The decision is recorded in two places -- the flag that gates
        ``_fit_model`` and the ``results_metadata`` entry that reaches the
        checkpoint and the run report -- so every path that gives up on
        ``train_Yvar`` goes through here rather than setting them separately.
        """
        warnings.warn(reason, UserWarning, stacklevel=stacklevel)
        self._mc_uncertainty_fallback = True
        self.results_metadata["mc_uncertainty_fallback"] = True
        self.train_Yvar = None

    def _mobo_ckpt_path(self) -> str:
        if self._mobo_checkpoint_loc is None:
            log_dir = Path(self.BaseDirectory) / self.SimulationName / "logs"
            self._mobo_checkpoint_loc = str(log_dir / "MOBOCheckpoint.npz")
        return self._mobo_checkpoint_loc

    # Constraints (g(x) <= 0 feasible, pymoo/TopasMOO convention)

    def _has_constraints(self) -> bool:
        """Whether any decision constraint is configured."""
        return bool(self._decision_constraints)

    @staticmethod
    def _constraint_scalar(value: Any, source: str) -> float:
        """Convert one decision-constraint result to a finite scalar."""
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        arr = np.asarray(value, dtype=float)
        if arr.size != 1 or not np.all(np.isfinite(arr)):
            raise InvalidParameterError(
                f"{source} must return one finite scalar per decision vector; "
                f"got shape {arr.shape}."
            )
        return float(arr.reshape(-1)[0])

    def _observed_feasible_mask(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Return decision-constraint feasibility for a validated observation prefix."""
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        n = len(Y)
        if X.shape != (n, len(self.ParameterNames)) or not np.all(np.isfinite(X)):
            raise InvalidParameterError(
                "Decision observations are missing, non-finite, or desynchronized."
            )
        if Y.shape != (n, self.n_objectives) or not np.all(np.isfinite(Y)):
            raise InvalidParameterError(
                "Objective observations are missing, non-finite, or desynchronized."
            )
        mask = np.ones(n, dtype=bool)
        if self._decision_constraints:
            torch = self._ensure_botorch()["torch"]
            for i, x in enumerate(X):
                x_t = torch.as_tensor(x, dtype=torch.double)
                for j, constraint in enumerate(self._decision_constraints):
                    value = self._constraint_scalar(
                        constraint(x_t),
                        f"decision constraint {j}",
                    )
                    if value > 0:
                        mask[i] = False
                        break
        return mask

    def feasible_mask(self) -> np.ndarray:
        """Mask observations satisfying every ``decision_constraints`` entry (``g <= 0``).

        All-``True`` when no decision constraints are configured. Missing or
        desynchronized observations raise instead of treating unknown
        feasibility as true.

        Feasibility is row-independent and observations are append-only, so the
        answer is cached, and only rows added since the last call are scored.
        Without that, a single tell evaluated every user constraint callable
        over the entire history several times over.
        """
        n = 0 if self.train_Y is None else len(self.train_Y)
        if n == 0:
            return np.ones(0, dtype=bool)
        if self.train_X is None:
            raise InvalidParameterError("Stored decision observations are missing.")
        cached = self._feasible_cache
        if cached is not None and len(cached) == n:
            return cached
        if cached is not None and len(cached) < n:
            start = len(cached)
            new_rows = self._observed_feasible_mask(
                self.train_X[start:],
                self.train_Y[start:],
            )
            mask = np.concatenate([cached, new_rows])
        else:
            mask = self._observed_feasible_mask(self.train_X, self.train_Y)
        self._feasible_cache = mask
        return mask

    def _feasible_prefix(self, n_rows: int) -> np.ndarray:
        """Feasibility for the first ``n_rows`` committed observations.

        A slice of the full mask: whether a row is feasible does not depend on
        which other rows are present.
        """
        return self.feasible_mask()[:n_rows]

    def _failed_prefix(self, n_rows: int) -> np.ndarray:
        """Failure mask for the first ``n_rows`` observations (all-``False`` if unset)."""
        if self.train_failed is None:
            return np.zeros(n_rows, dtype=bool)
        return np.asarray(self.train_failed[:n_rows], dtype=bool)

    def eligible_mask(self) -> np.ndarray:
        """Observations that may be reported or used to derive a reference point.

        An observation is eligible when it satisfies every configured constraint
        **and** did not come from a failed evaluation. ``feasible_mask`` answers
        only the constraint question; this is the mask every consumer wants,
        because a penalized failure carries ``failure_penalty`` (``1e6`` by
        default) in every objective and would otherwise define the nadir.
        """
        feasible = self.feasible_mask()
        return feasible & ~self._failed_prefix(len(feasible))

    def _eligible_rows(self, n_rows: int) -> np.ndarray:
        """Reportable evaluations for the base class's running Pareto front.

        Without this override the mid-run ``ParetoFront_Running.txt`` and the
        intermediate convergence plots would show infeasible designs and
        ``failure_penalty`` rows -- exactly what the quarantine removes from the
        final front -- so a long campaign displayed a monitoring front that
        contradicted the result it ended on.

        ``AllObjectiveFunctionValues`` grows once per *evaluation* while
        ``train_Y`` grows once per *batch*, so this is asked about rows that
        have not been told yet. The two answers are sourced accordingly.
        """
        mask = np.ones(n_rows, dtype=bool)
        # Failure is looked up per design
        for i in range(min(n_rows, len(self.AllDecisionVariables))):
            if self._eval_failed.get(self._cache_key(self.AllDecisionVariables[i]), False):
                mask[i] = False
        # Constraint feasibility is positional, and a design's constraint values
        # only reach the optimizer at tell(). Apply it only when the two
        # histories are known to be row-aligned, which tell() guarantees, since
        # _update_pareto_attrs rewrites AllObjectiveFunctionValues from train_Y.
        if self.train_Y is not None and n_rows == len(self.train_Y):
            mask &= self.eligible_mask()
        return mask

    def _gp_training_rows(self) -> np.ndarray:
        """Row indices the GPs are fitted on: everything except failed evaluations.

        A penalized failure carries ``failure_penalty`` in every objective, which
        would dominate ``Standardize``'s mean/std and flatten the posterior over
        the region that actually matters. Failures are therefore dropped from the
        training set (the base-class evaluation cache still prevents TOPAS from
        re-running them). If fewer than two usable rows remain, every row is kept
        so the run continues instead of dying on an unfittable model.
        """
        assert self.train_Y is not None
        n = len(self.train_Y)
        keep = np.where(~self._failed_prefix(n))[0]
        if len(keep) < 2:
            if n and len(keep) < n:
                warnings.warn(
                    f"{n - len(keep)} of {n} observations are failed evaluations; "
                    "too few usable rows remain to fit the GP on successes alone, "
                    "so the penalized rows are being included. Expect a poor "
                    "surrogate until real evaluations succeed.",
                    UserWarning,
                    stacklevel=3,
                )
            return np.arange(n)
        return keep

    def _fit_model(self):
        bt = self._ensure_botorch()
        torch = bt["torch"]
        assert self.train_X is not None and self.train_Y is not None
        rows = self._gp_training_rows()
        train_X = self._torch_X(np.asarray(self.train_X)[rows])
        train_Y = self._torch_Y_botorch(np.asarray(self.train_Y)[rows])
        d = train_X.shape[-1]
        m = train_Y.shape[-1]

        train_Yvar = None
        if self.train_Yvar is not None and not self._mc_uncertainty_fallback:
            if (
                np.shape(self.train_Yvar) != np.shape(self.train_Y)
                or not np.all(np.isfinite(self.train_Yvar))
                or np.any(np.asarray(self.train_Yvar) < 0)
            ):
                self._fall_back_to_inferred_noise(
                    f"train_Yvar shape/values {np.shape(self.train_Yvar)} are "
                    f"incompatible with train_Y shape {np.shape(self.train_Y)}; "
                    "falling back to inferred noise for the whole run.",
                    stacklevel=3,
                )
            else:
                train_Yvar = torch.as_tensor(
                    self._to_botorch_variance(np.asarray(self.train_Yvar)[rows]),
                    dtype=torch.double,
                )

        def _gp(target, yvar=None):
            return bt["SingleTaskGP"](
                train_X,
                target,
                train_Yvar=yvar,
                input_transform=bt["Normalize"](d=d, bounds=self._bounds_tensor),
                outcome_transform=bt["Standardize"](m=1),
            )

        # One GP per objective; the model's only outputs are the objectives, so
        # BoTorch's default identity objective is already correct downstream.
        models = []
        for j in range(m):
            yvar_j = None if train_Yvar is None else train_Yvar[:, j : j + 1]
            models.append(_gp(train_Y[:, j : j + 1], yvar_j))

        model = bt["ModelListGP"](*models)
        mll = bt["SumMarginalLogLikelihood"](model.likelihood, model)
        bt["fit_gpytorch_mll"](mll)
        self.model = model
        return model

    def _acqf_ref_point_botorch(self):
        """Reference point for acquisition in BoTorch (maximize) space.

        When no user ``ref_point`` is given, the reference is the anti-ideal
        point (component-wise worst observed value) across all eligible
        observations, padded by 10 % of the per-objective range.  This follows
        the recommendation of Ishibuchi et al. (2018): using the Pareto-front
        nadir instead biases the search toward the current front and prevents
        the acquisition from rewarding exploration beyond it.

            Ishibuchi, H., Imada, R., Setoguchi, Y., & Nojima, Y. (2018).
            Reference Point Specification in Hypervolume Calculation for Fair
            Comparison and Efficient Search.  *Proc. GECCO*, pp. 585--592.

        Eligible observations exclude infeasible designs and penalized
        failures: an infeasible outlier would drag the reference into a
        region the acquisition is not allowed to exploit, and a penalized
        failure carries ``failure_penalty`` in every objective which would
        push the reference out by orders of magnitude.  Falls back to all
        observations while nothing eligible exists yet.
        """
        bt = self._ensure_botorch()
        torch = bt["torch"]
        assert self.train_Y is not None
        Y_ref_src = np.asarray(self.train_Y, dtype=float)
        eligible = self.eligible_mask()
        if eligible.any():
            Y_ref_src = Y_ref_src[eligible]
        Y_max = torch.as_tensor(self._to_botorch_objectives(Y_ref_src), dtype=torch.double)

        if self._user_ref_point_min is not None:
            user_max = torch.as_tensor(
                self._to_botorch_objectives(self._user_ref_point_min),
                dtype=torch.double,
            )
            # In maximize space, ref should be dominated by nadir of observations
            # (i.e. componentwise worse / smaller than observed max-space nadir).
            nadir_max = Y_max.min(dim=0).values
            if torch.any(user_max > nadir_max):
                warnings.warn(
                    f"User ref_point (minimize space) {self._user_ref_point_min.tolist()} "
                    f"is not dominated by the current nadir "
                    f"{(-nadir_max).detach().cpu().numpy().tolist()} in minimization "
                    "space; acquisition quality may degrade.",
                    UserWarning,
                    stacklevel=2,
                )
            return user_max

        return bt["infer_reference_point"](Y_max)

    # Shared margin formula with NSGAII_Optimizer
    # MOBO feeds eligible observations into that formula, NSGA-II feeds each
    # generation's population. Both report one HypervolumeHistory entry per
    # algorithm step.
    _reference_from_objectives = staticmethod(hypervolume_reference_point)

    def _history_prefix_lengths(self) -> list[int]:
        """Cumulative observation counts represented by ``PopulationHistory``."""
        lengths = list(
            itertools.accumulate(
                len(np.atleast_2d(population)) for _batch, population in self.PopulationHistory
            )
        )
        total = lengths[-1] if lengths else 0
        if self.train_Y is not None and total != len(self.train_Y):
            raise InvalidParameterError(
                "PopulationHistory is desynchronized from the stored observations: "
                f"history covers {total} rows but train_Y has {len(self.train_Y)}."
            )
        return lengths

    def _eligible_nd_indices(self, n_rows: int) -> np.ndarray:
        """Rows of the eligible non-dominated observations in a prefix.

        Eligible means feasible under every configured constraint *and* not a
        penalized failure. An empty result means nothing was eligible at all,
        since the non-dominated front of a non-empty set is never empty.
        """
        assert self.train_X is not None and self.train_Y is not None
        Y = np.asarray(self.train_Y[:n_rows], dtype=float)
        mask = self._feasible_prefix(n_rows) & ~self._failed_prefix(n_rows)
        eligible = np.where(mask)[0]
        if len(eligible) == 0:
            return eligible
        bt = self._ensure_botorch()
        torch = bt["torch"]

        nd_mask = bt["is_non_dominated"](
            torch.as_tensor(Y[eligible], dtype=torch.double),
            maximize=False,
            deduplicate=False,
        )
        return eligible[nd_mask.detach().cpu().numpy()]

    def _nd_front_at_prefix(self, n_rows: int) -> np.ndarray:
        """Eligible non-dominated objectives among the first ``n_rows`` observations.

        Memoized: the front depends only on the observations, never on the
        hypervolume reference, and observations are append-only, so a prefix's
        front never changes once computed.
        ``_update_pareto_attrs`` seeds the entry for the full history, so the
        common path sorts once per ``tell`` rather than twice.
        """
        cached = self._nd_front_cache.get(n_rows)
        if cached is not None:
            return cached
        nd_idx = self._eligible_nd_indices(n_rows)
        front = (
            np.asarray(self.train_Y, dtype=float)[nd_idx]
            if len(nd_idx)
            else np.empty((0, self.n_objectives))
        )
        self._nd_front_cache[n_rows] = front
        return front

    def _hypervolume_at_prefix(self, n_rows: int, indicator: Any = None) -> float:
        """Hypervolume of the eligible non-dominated observations in one prefix.

        :param indicator: Optional pre-built ``HV`` for the current reference,
            so a full-history recompute constructs one indicator instead of one
            per prefix.
        """
        if self._hv_ref_fixed is None:
            return 0.0
        nd = self._nd_front_at_prefix(n_rows)
        if len(nd) == 0:
            return 0.0
        try:
            if indicator is None:
                indicator = HV(ref_point=self._hv_ref_fixed)
            return float(indicator(nd))
        except Exception as exc:
            logger.warning("Could not compute hypervolume: %s", exc)
            return 0.0

    def _recompute_hv_history(self) -> None:
        """Recompute every history entry under the current reporting reference."""
        indicator = None if self._hv_ref_fixed is None else HV(ref_point=self._hv_ref_fixed)
        self.HypervolumeHistory = [
            self._hypervolume_at_prefix(n_rows, indicator)
            for n_rows in self._history_prefix_lengths()
        ]

    def _update_hv_history(self) -> float:
        """Update comparable eligible hypervolume history.

        One entry is appended (or the full series recomputed) per ``tell``,
        matching ``NSGAII_Optimizer``'s one-entry-per-generation cadence on
        ``HypervolumeHistory``. A user reference point remains fixed. Otherwise
        the inferred reference is based only on *eligible* observations
        (feasible, and not a penalized failure), expands componentwise whenever
        newly observed eligible objectives require it, and all prior entries are
        recomputed so the series never mixes references. Until the first
        eligible observation, hypervolume is zero and no inferred reference is
        committed.
        """
        assert self.train_X is not None and self.train_Y is not None
        if self._user_ref_point_min is not None:
            candidate = self._user_ref_point_min.copy()
        else:
            eligible = self.eligible_mask()
            candidate = (
                self._reference_from_objectives(self.train_Y[eligible]) if eligible.any() else None
            )

        previous = self._hv_ref_fixed
        if candidate is not None:
            if previous is None:
                self._hv_ref_fixed = candidate
            elif self._user_ref_point_min is None:
                self._hv_ref_fixed = np.maximum(previous, candidate)

        reference_changed = self._hv_ref_fixed is not None and (
            previous is None or not np.array_equal(previous, self._hv_ref_fixed)
        )
        if reference_changed:
            self._recompute_hv_history()
        else:
            self.HypervolumeHistory.append(self._hypervolume_at_prefix(len(self.train_Y)))
        return self.HypervolumeHistory[-1]

    def _update_pareto_attrs(self) -> None:
        """Refresh the reported Pareto set from the observations.

        With constraints, only feasible observations are eligible: an
        infeasible design must never be reported as an optimum, even if it
        dominates every feasible one in objective space. Penalized failures are
        likewise excluded -- reporting a crashed evaluation as an optimum would
        be worse than reporting nothing.

        Also rewrites ``ParetoFront_Running.txt`` so the mid-run monitoring
        front uses the same eligibility definition as the official final front
        written to ``ParetoFront.txt`` (matching the base-class split used by
        ``NSGAII_Optimizer``).
        """
        assert self.train_X is not None and self.train_Y is not None
        Y = np.asarray(self.train_Y, dtype=float)
        X = np.asarray(self.train_X, dtype=float)
        nd_idx = self._eligible_nd_indices(len(Y))
        if len(nd_idx) == 0:
            if self._has_constraints() or self._failed_prefix(len(Y)).any():
                logger.warning(
                    "No eligible observations yet (all %d are infeasible or "
                    "failed evaluations); the Pareto front is empty.",
                    len(Y),
                )
            self.ParetoObjectives = np.empty((0, self.n_objectives))
            self.ParetoDecisionVars = np.empty((0, X.shape[1]))
        else:
            self.ParetoObjectives = np.atleast_2d(Y[nd_idx])
            self.ParetoDecisionVars = np.atleast_2d(X[nd_idx])

        self._nd_front_cache.setdefault(len(Y), self.ParetoObjectives)
        self.AllObjectiveFunctionValues = list(Y)
        self.AllDecisionVariables = list(X)

        try:
            LogParetoFrontToFile(
                self._ParetoRunningLogFileLoc,
                self.ParetoObjectives,
                self.ParameterNames,
                self.n_objectives,
                ParetoDecisionVars=self.ParetoDecisionVars,
            )
        except OSError as exc:
            logger.warning("Could not write running Pareto front: %s", exc)

    def _botorch_decision_constraints(self) -> list[tuple[Callable, bool]] | None:
        """Wrap ``decision_constraints`` for ``optimize_acqf``.

        User callables follow TopasMOO / pymoo convention ``g(X) <= 0`` feasible.
        BoTorch expects ``callable(x) >= 0``, so we negate. Intra-point constraints
        (``True``) receive a 1-D ``d``-vector.
        """
        if not self._decision_constraints:
            return None

        wrapped: list[tuple[Callable, bool]] = []
        for g in self._decision_constraints:

            def _ge0(x, g=g):
                # BoTorch intra-point: x is 1-D (d,). User g may accept batch dims.
                val = g(x)
                if hasattr(val, "numel"):
                    if val.numel() != 1:
                        raise InvalidParameterError(
                            "Each decision constraint must return one scalar."
                        )
                    return -val.reshape(())
                arr = np.asarray(val)
                if arr.size != 1:
                    raise InvalidParameterError("Each decision constraint must return one scalar.")
                return -float(arr.reshape(-1)[0])

            wrapped.append((_ge0, True))
        return wrapped

    def _rejection_sample_feasible(self, n: int, rng, what: str) -> np.ndarray:
        """Uniformly draw ``n`` points in bounds satisfying every decision constraint.

        :param what: Phrase naming the points, used in the failure message.
        :raises RuntimeError: If ``10_000 * n`` draws do not yield ``n`` feasible
            points, i.e. the feasible region is too small to sample blindly.
        """
        torch = self._ensure_botorch()["torch"]
        cons = self._botorch_decision_constraints() or []
        lower = self._bounds_tensor[0].detach().cpu().numpy()
        upper = self._bounds_tensor[1].detach().cpu().numpy()
        rows: list[np.ndarray] = []
        max_tries = 10_000 * n
        tries = 0
        while len(rows) < n and tries < max_tries:
            tries += 1
            x_np = lower + rng.random(len(lower)) * (upper - lower)
            x_t = torch.as_tensor(x_np, dtype=torch.double)
            # Wrapped constraints follow BoTorch's ">= 0 is feasible" convention.
            if all(float(fn(x_t)) >= 0 for fn, _intra in cons):
                rows.append(x_np)
        if len(rows) < n:
            raise RuntimeError(
                f"Could only sample {len(rows)}/{n} feasible {what} after {tries} tries."
            )
        return np.asarray(rows, dtype=float)

    def _sample_feasible_initial_conditions(self, q: int):
        """Rejection-sample ``(num_restarts, q, d)`` starts satisfying decision constraints."""
        torch = self._ensure_botorch()["torch"]
        rng = np.random.default_rng(
            None if self.seed is None else self.seed + 17 * self._batch_index
        )
        rows = self._rejection_sample_feasible(
            self.num_restarts * q,
            rng,
            "acquisition initial conditions",
        )
        starts = rows.reshape(self.num_restarts, q, self._bounds_tensor.shape[-1])
        return torch.as_tensor(starts, dtype=torch.double)

    def _optimize_acqf(self, acqf, *, q: int, sequential: bool, nl_constraints):
        """Maximize ``acqf`` over the bounds, returning ``(q, d)`` candidates.

        With decision constraints BoTorch cannot generate its own starts (its
        heuristic ignores nonlinear constraints), so feasible
        ``batch_initial_conditions`` are supplied instead of ``raw_samples``.
        """
        kwargs: dict[str, Any] = {
            "acq_function": acqf,
            "bounds": self._bounds_tensor,
            "q": q,
            "num_restarts": self.num_restarts,
            "raw_samples": self.raw_samples,
            "sequential": sequential,
        }
        if nl_constraints:
            kwargs["nonlinear_inequality_constraints"] = nl_constraints
            kwargs["batch_initial_conditions"] = self._sample_feasible_initial_conditions(q=q)
            kwargs["raw_samples"] = None
        candidates, _ = self._ensure_botorch()["optimize_acqf"](**kwargs)
        return candidates

    # Checkpointing

    def save_checkpoint(self) -> None:
        """Write versioned MOBO state to ``logs/MOBOCheckpoint.npz``."""
        path = self._mobo_ckpt_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "n_objectives": np.asarray([self.n_objectives]),
            "n_parameters": np.asarray([len(self.ParameterNames)]),
            "parameter_names": np.asarray(self.ParameterNames, dtype=str),
            "lower_bounds": np.asarray(self.LowerBounds, dtype=float),
            "upper_bounds": np.asarray(self.UpperBounds, dtype=float),
            "train_X": np.asarray(self.train_X, dtype=float)
            if self.train_X is not None
            else np.zeros((0, 0)),
            "train_Y": np.asarray(self.train_Y, dtype=float)
            if self.train_Y is not None
            else np.zeros((0, 0)),
            "HypervolumeHistory": np.asarray(self.HypervolumeHistory, dtype=float),
            "batch_index": np.asarray([self._batch_index]),
            "mc_uncertainty_fallback": np.asarray([int(self._mc_uncertainty_fallback)]),
            "results_metadata_json": np.asarray(json.dumps(self.results_metadata)),
        }
        if self._hv_ref_fixed is not None:
            payload["hv_ref_fixed"] = np.asarray(self._hv_ref_fixed, dtype=float)

        if self.PopulationHistory:
            payload["PopulationHistoryBatch"] = np.concatenate(
                [
                    np.full(len(np.atleast_2d(pop)), int(b), dtype=np.int64)
                    for b, pop in self.PopulationHistory
                ]
            )
            payload["PopulationHistoryY"] = np.vstack(
                [np.atleast_2d(np.asarray(pop, dtype=float)) for _, pop in self.PopulationHistory]
            )
        # Which rows are penalized failures. Written whenever there are
        # observations, not only when something failed, so the mask stays aligned
        # with train_X/train_Y; an absent key means there are no observations.
        if self.train_failed is not None and len(self.train_failed):
            payload["train_failed"] = np.asarray(self.train_failed, dtype=bool).astype(np.int8)
        if self.train_Yvar is not None:
            payload["train_Yvar"] = np.asarray(self.train_Yvar, dtype=float)
        if self._pending_X is not None:
            payload["pending_X"] = np.asarray(self._pending_X, dtype=float)
        if self.ParetoObjectives is not None and len(self.ParetoObjectives):
            payload["ParetoObjectives"] = np.asarray(self.ParetoObjectives, dtype=float)
        if self.ParetoDecisionVars is not None and len(np.atleast_1d(self.ParetoDecisionVars)):
            payload["ParetoDecisionVars"] = np.asarray(self.ParetoDecisionVars, dtype=float)

        checkpoint_tmp = path + ".tmp.npz"
        np.savez_compressed(checkpoint_tmp, **payload)
        os.replace(checkpoint_tmp, path)

        if not self._meta_written:
            meta = {
                "batch_size": self.batch_size,
                "n_init": self.n_init,
                "acquisition": self._acquisition_resolved,
                "seed": self.seed,
                "n_objectives": self.n_objectives,
                "parameter_names": list(self.ParameterNames),
            }
            meta_path = path + ".meta.json"
            meta_tmp = meta_path + ".tmp"
            with open(meta_tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            os.replace(meta_tmp, meta_path)
            self._meta_written = True

    def _validate_checkpoint_problem(
        self,
        arrays: dict[str, np.ndarray],
    ) -> None:
        """Reject checkpoint data from a different optimization problem."""
        d = len(self.ParameterNames)
        expected_scalars = {
            "n_objectives": self.n_objectives,
            "n_parameters": d,
        }
        for key, expected in expected_scalars.items():
            if key not in arrays or int(arrays[key][0]) != expected:
                got = None if key not in arrays else int(arrays[key][0])
                raise InvalidParameterError(
                    f"MOBO checkpoint {key}={got} is incompatible with "
                    f"the current value {expected}."
                )
        stored_names = np.asarray(arrays.get("parameter_names", [])).tolist()
        if stored_names != list(self.ParameterNames):
            raise InvalidParameterError(
                "MOBO checkpoint parameter names do not match the current problem."
            )
        for key, current in (
            ("lower_bounds", self.LowerBounds),
            ("upper_bounds", self.UpperBounds),
        ):
            if key not in arrays or not np.array_equal(
                np.asarray(arrays[key], dtype=float),
                np.asarray(current, dtype=float),
            ):
                raise InvalidParameterError(
                    f"MOBO checkpoint {key} do not match the current problem."
                )

        X = np.asarray(arrays["train_X"], dtype=float)
        Y = np.asarray(arrays["train_Y"], dtype=float)
        if X.size == 0:
            if Y.size != 0:
                raise InvalidParameterError(
                    "MOBO checkpoint has objective rows without decision rows."
                )
        elif (
            X.ndim != 2
            or X.shape[1] != d
            or Y.shape != (len(X), self.n_objectives)
            or not np.all(np.isfinite(X))
            or not np.all(np.isfinite(Y))
        ):
            raise InvalidParameterError(
                "MOBO checkpoint train_X/train_Y shapes or values are "
                "incompatible with the current problem."
            )

    def _decode_population_history(
        self,
        arrays: dict[str, np.ndarray],
    ) -> list[tuple[int, np.ndarray]]:
        """Decode the run-length-encoded per-batch objective history."""
        assert self.train_Y is not None
        has_batches = "PopulationHistoryBatch" in arrays
        has_values = "PopulationHistoryY" in arrays
        history: list[tuple[int, np.ndarray]] = []
        if has_batches != has_values:
            raise InvalidParameterError("MOBO checkpoint contains only half of PopulationHistory.")
        if has_batches:
            batches = np.asarray(arrays["PopulationHistoryBatch"]).reshape(-1)
            pops = np.atleast_2d(np.asarray(arrays["PopulationHistoryY"], dtype=float))
            if (
                    len(batches) != len(pops)
                    or pops.shape[1] != self.n_objectives
                    or not np.all(np.isfinite(pops))
            ):
                raise InvalidParameterError(
                    "MOBO checkpoint PopulationHistory arrays are malformed."
                )
            if len(batches):
                boundaries = np.flatnonzero(np.diff(batches)) + 1
                for chunk in np.split(np.arange(len(batches)), boundaries):
                    history.append((int(batches[chunk[0]]), pops[chunk].copy()))

        if sum(len(pop) for _, pop in history) != len(self.train_Y):
            raise InvalidParameterError(
                "MOBO checkpoint PopulationHistory does not cover every observation."
            )
        if len(history) != self._batch_index:
            raise InvalidParameterError(
                "MOBO checkpoint batch_index does not match PopulationHistory."
            )
        if [batch for batch, _ in history] != list(range(self._batch_index)):
            raise InvalidParameterError(
                "MOBO checkpoint PopulationHistory batch labels are not consecutive."
            )
        return history

    def _restore_algorithm_state(self) -> None:
        """Resume: reload the MOBO checkpoint alongside the base state.

        Fired by ``SetUpDirectoryStructure`` when ``resume=True``, so the
        observation history, hypervolume reference, and batch index come back at
        the same point in the lifecycle as the evaluation cache rather than
        later from ``run()``.
        """
        self.load_checkpoint()

    def load_checkpoint(self) -> bool:
        """Load and validate ``MOBOCheckpoint.npz`` if present."""
        path = self._mobo_ckpt_path()
        if not os.path.isfile(path):
            return False
        try:
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: np.asarray(data[key]).copy() for key in data.files}
        except (OSError, ValueError, KeyError) as exc:
            raise InvalidParameterError(f"Could not read MOBO checkpoint {path}: {exc}") from exc
        for required in (
            "train_X",
            "train_Y",
            "HypervolumeHistory",
            "batch_index",
            "mc_uncertainty_fallback",
        ):
            if required not in arrays:
                raise InvalidParameterError(
                    f"MOBO checkpoint is missing required key {required!r}."
                )
        self._validate_checkpoint_problem(arrays)

        self.train_X = np.asarray(arrays["train_X"], dtype=float)
        self.train_Y = np.asarray(arrays["train_Y"], dtype=float)
        if self.train_X.size == 0:
            self.train_X = None
            self.train_Y = None
        self.HypervolumeHistory = [
            float(v) for v in np.asarray(arrays["HypervolumeHistory"]).reshape(-1)
        ]
        self._batch_index = int(arrays["batch_index"][0])
        self._mc_uncertainty_fallback = bool(arrays["mc_uncertainty_fallback"][0])
        if "results_metadata_json" in arrays:
            try:
                restored_metadata = json.loads(str(arrays["results_metadata_json"].item()))
                if isinstance(restored_metadata, dict):
                    self.results_metadata = restored_metadata
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.warning("Ignoring malformed results_metadata in checkpoint.")
        if self._mc_uncertainty_fallback:
            self.results_metadata["mc_uncertainty_fallback"] = True

        self._nd_front_cache = {}
        # The history was just replaced rather than appended to.
        self._feasible_cache = None
        self._hv_ref_fixed = (
            np.asarray(arrays["hv_ref_fixed"], dtype=float) if "hv_ref_fixed" in arrays else None
        )
        if self._user_ref_point_min is not None:
            # An explicit ref_point on the resumed run is a deliberate instruction
            # and must beat whatever the previous run stored, otherwise the
            # acquisition would use the new reference while HypervolumeHistory
            # kept reporting against the old one -- and, because the expansion
            # branch is skipped whenever a user reference is set, the stale value
            # could never grow again either.
            stale = self._hv_ref_fixed
            self._hv_ref_fixed = self._user_ref_point_min.copy()
            if stale is not None and not np.array_equal(
                np.asarray(stale, dtype=float), self._hv_ref_fixed
            ):
                warnings.warn(
                    f"Checkpoint hypervolume reference {np.asarray(stale).tolist()} "
                    f"differs from the ref_point {self._hv_ref_fixed.tolist()} given "
                    "to this run; using ref_point and recomputing the history so "
                    "acquisition and reporting agree.",
                    UserWarning,
                    stacklevel=2,
                )
        n_obs = 0 if self.train_Y is None else len(self.train_Y)
        if "train_failed" in arrays:
            restored_failed = np.asarray(arrays["train_failed"]).reshape(-1).astype(bool)
            if len(restored_failed) != n_obs:
                raise InvalidParameterError(
                    f"MOBO checkpoint train_failed has {len(restored_failed)} entries "
                    f"but the checkpoint holds {n_obs} observations."
                )
            self.train_failed = restored_failed
        else:
            # save_checkpoint only omits the key when there are no observations
            # to mark, so an absent mask means nothing was recorded as failed.
            self.train_failed = np.zeros(n_obs, dtype=bool) if n_obs else None

        if "train_Yvar" in arrays:
            restored_yvar = np.asarray(arrays["train_Yvar"], dtype=float)
            invalid_values = not np.all(np.isfinite(restored_yvar)) or np.any(restored_yvar < 0)
            if self.train_Y is None and not invalid_values:
                # A checkpoint written by ask() before the first tell() can
                # contain constructor-supplied variances for that pending
                # initial design. Preserve them for row-by-row validation when
                # the pending observations are eventually told.
                self._constructor_train_Yvar = np.atleast_2d(restored_yvar)
                self.train_Yvar = self._constructor_train_Yvar
            elif (
                self.train_Y is None or restored_yvar.shape != self.train_Y.shape or invalid_values
            ):
                self._fall_back_to_inferred_noise(
                    "Checkpoint train_Yvar is invalid; falling back to inferred "
                    "noise for the whole run."
                )
            else:
                self.train_Yvar = restored_yvar
        elif self.train_Y is not None:
            self.train_Yvar = None
        if self._mc_uncertainty_fallback:
            self.train_Yvar = None

        self._pending_X = (
            np.atleast_2d(np.asarray(arrays["pending_X"], dtype=float))
            if "pending_X" in arrays
            else None
        )
        if self._pending_X is not None:
            if self._pending_X.shape[1] != len(self.ParameterNames) or not np.all(
                np.isfinite(self._pending_X)
            ):
                raise InvalidParameterError("MOBO checkpoint pending_X is malformed.")

        if self.train_Y is not None:
            self.PopulationHistory = self._decode_population_history(arrays)
            if self._hv_ref_fixed is None:
                if self._user_ref_point_min is not None:
                    self._hv_ref_fixed = self._user_ref_point_min.copy()
                else:
                    feasible = self.feasible_mask()
                    if feasible.any():
                        self._hv_ref_fixed = self._reference_from_objectives(self.train_Y[feasible])
            if self._hv_ref_fixed is not None and (
                self._hv_ref_fixed.shape != (self.n_objectives,)
                or not np.all(np.isfinite(self._hv_ref_fixed))
            ):
                raise InvalidParameterError("MOBO checkpoint hypervolume reference is malformed.")
            self._recompute_hv_history()
            self._update_pareto_attrs()
            if self.train_Yvar is not None:
                self._variance_cache = {
                    self._cache_key(x): np.asarray(yvar, dtype=float).copy()
                    for x, yvar in zip(
                        self.train_X,
                        self.train_Yvar,
                        strict=True,
                    )
                }
        else:
            self.PopulationHistory = []
            self.HypervolumeHistory = []
        logger.info(
            "Loaded MOBO checkpoint (%s observations, batch_index=%s)",
            0 if self.train_X is None else len(self.train_X),
            self._batch_index,
        )
        return True


    # Ask / tell / run

    def _inject_start_point(self, X: np.ndarray) -> np.ndarray:
        """Replace the first initial-design row with the user's ``start_point``.

        Mirrors ``NSGAII_Optimizer``'s ``_StartPointSampling`` so a known-good
        design is actually evaluated rather than merely validated. Skipped (with
        a warning) when it would violate a decision constraint, since the rest
        of the initial design is rejection-sampled to be feasible.
        """
        if not self.include_start_point or len(X) == 0:
            return X
        start = np.asarray(self.StartingValues, dtype=float).reshape(-1)
        if start.shape[0] != X.shape[1]:
            return X
        if self._decision_constraints:
            torch = self._ensure_botorch()["torch"]
            x_t = torch.as_tensor(start, dtype=torch.double)
            cons = self._botorch_decision_constraints() or []
            if not all(float(fn(x_t)) >= 0 for fn, _ in cons):
                warnings.warn(
                    "start_point violates a decision constraint and was not "
                    "added to the initial design.",
                    UserWarning,
                    stacklevel=3,
                )
                return X
        X = X.copy()
        X[0] = start
        logger.info("Injected start_point as the first initial-design point.")
        return X

    def ask(self) -> np.ndarray:
        """Propose the next batch of candidates, shape ``(batch_size, d)``.

        On an empty history, draws an initial design of size ``n_init``
        (returned in full; callers should evaluate all rows before ``tell``):
        a Sobol sequence normally, or uniform rejection sampling when
        ``decision_constraints`` are set, since a Sobol design gives no
        feasibility guarantee. Unless ``include_start_point=False``, the first
        row is replaced by the user's ``start_point``. After initialization,
        returns ``batch_size`` candidates from acquisition optimization.
        """
        self.SetUpDirectoryStructure()
        bt = self._ensure_botorch()
        torch = bt["torch"]
        if self._pending_X is not None:
            return self._pending_X.copy()
        if self.seed is not None:
            # Derive acquisition randomness from persisted state so resuming at
            # a batch boundary matches an uninterrupted run.
            torch.manual_seed(self.seed + 104_729 * self._batch_index)

        if self.train_X is None or len(self.train_X) == 0:
            n = self.n_init
            if self._decision_constraints:
                # A Sobol design gives no feasibility guarantee, so fall back to
                # uniform rejection sampling inside the feasible region.
                X = self._rejection_sample_feasible(
                    n,
                    np.random.default_rng(self.seed),
                    "initial design points",
                )
            else:
                samples = bt["draw_sobol_samples"](
                    bounds=self._bounds_tensor,
                    n=1,
                    q=n,
                    seed=self.seed,
                )
                # draw_sobol_samples → shape (n=1, q=n_init, d) for recent BoTorch
                X = samples.squeeze(0).detach().cpu().numpy()
            X = self._inject_start_point(np.asarray(X, dtype=float))
            self._pending_X = np.asarray(X, dtype=float)
            try:
                self.save_checkpoint()
            except (OSError, ValueError) as exc:
                logger.warning("Could not checkpoint pending initial design: %s", exc)
            return self._pending_X.copy()

        t0 = time.perf_counter()
        model = self._fit_model()
        fit_s = time.perf_counter() - t0

        # X_baseline must be the points the model was actually fitted on:
        # penalized failures are quarantined out of the training set, so feeding
        # them to the acquisition as "already observed candidates for best" would
        # reintroduce the sentinel the quarantine exists to remove.
        gp_rows = self._gp_training_rows()
        train_X = self._torch_X(np.asarray(self.train_X)[gp_rows])
        train_Y_b = self._torch_Y_botorch(np.asarray(self.train_Y)[gp_rows])
        ref = self._acqf_ref_point_botorch()
        sampler = bt["SobolQMCNormalSampler"](sample_shape=torch.Size([128]), seed=self.seed)
        nl_constraints = self._botorch_decision_constraints()

        t1 = time.perf_counter()
        if self._acquisition_resolved == "qlognparego":
            # Redraw Chebyshev weights once per candidate within the batch
            # (spec requirement). Optimize q=1 sequentially with X_pending.
            from botorch.utils.sampling import sample_simplex

            pieces = []
            pending = None
            m = train_Y_b.shape[-1]
            for i in range(self.batch_size):
                weights = sample_simplex(
                    d=m,
                    n=1,
                    seed=None if self.seed is None else self.seed + self._batch_index * 1000 + i,
                    device=train_X.device,
                    dtype=train_X.dtype,
                ).view(-1)
                acqf = build_acquisition(
                    "qlognparego",
                    model,
                    train_X,
                    ref_point=ref,
                    sampler=sampler,
                    scalarization_weights=weights,
                    X_pending=pending,
                    bt=bt,
                )
                cand = self._optimize_acqf(
                    acqf, q=1, sequential=False, nl_constraints=nl_constraints
                )
                pieces.append(cand)
                pending = cand if pending is None else torch.cat([pending, cand], dim=-2)
            candidates = torch.cat(pieces, dim=-2)
        else:
            acqf = build_acquisition(
                self._acquisition_resolved,
                model,
                train_X,
                ref_point=ref,
                sampler=sampler,
                bt=bt,
            )
            candidates = self._optimize_acqf(
                acqf,
                q=self.batch_size,
                sequential=self.sequential,
                nl_constraints=nl_constraints,
            )
        acqf_s = time.perf_counter() - t1
        X = candidates.detach().cpu().numpy()
        self._pending_X = np.asarray(X, dtype=float)
        try:
            self.save_checkpoint()
        except (OSError, ValueError) as exc:
            logger.warning("Could not checkpoint pending candidates: %s", exc)

        hv = self.HypervolumeHistory[-1] if self.HypervolumeHistory else float("nan")
        n_pareto = len(self.ParetoObjectives) if hasattr(self.ParetoObjectives, "__len__") else 0
        logger.info(
            "batch=%s hypervolume=%s n_pareto=%s gp_fit_s=%s acqf_opt_s=%s",
            self._batch_index,
            _tensor_to_float(hv) if hv == hv else hv,
            n_pareto,
            f"{fit_s:.4f}",
            f"{acqf_s:.4f}",
        )
        return self._pending_X.copy()

    def _use_constructor_train_Yvar(self, Y: np.ndarray) -> np.ndarray | None:
        """Validate and return the constructor-supplied ``train_Yvar`` for ``Y``.

        Returns ``None`` (and switches the run to inferred noise) when the
        supplied array does not line up row-for-row with the first batch of
        observations, rather than silently dropping it as an earlier version did.
        """
        ctor = self._constructor_train_Yvar
        if ctor.shape != Y.shape:
            self._fall_back_to_inferred_noise(
                f"Constructor train_Yvar has shape {ctor.shape} but the first "
                f"observations have shape {Y.shape}; falling back to inferred "
                "noise for the whole run.",
                stacklevel=4,
            )
            return None
        if not np.all(np.isfinite(ctor)) or np.any(ctor < 0):
            self._fall_back_to_inferred_noise(
                "Constructor train_Yvar must be finite and non-negative; "
                "falling back to inferred noise for the whole run.",
                stacklevel=4,
            )
            return None
        return ctor

    def tell(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Yvar: np.ndarray | None = None,
        failed: np.ndarray | None = None,
    ) -> None:
        """Incorporate observations ``X`` ``(n, d)``, ``Y`` ``(n, m)`` (minimize).

        :param Yvar: Optional observation variance in minimization space.
        :param failed: Optional ``(n,)`` boolean mask marking rows whose
            objective values are the ``failure_penalty`` sentinel rather than a
            real measurement. Those rows are kept (so indices and history stay
            aligned) but are excluded from GP training, from both reference
            points, and from the reported Pareto front. Defaults to all-``False``.
            ``run()`` fills this in automatically; supply it when driving
            ``ask``/``tell`` yourself and your evaluator can fail.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        d = len(self.ParameterNames)
        if X.shape[1] != d:
            raise InvalidParameterError(f"X must have {d} columns. Got {X.shape}.")
        if Y.shape[0] != X.shape[0]:
            raise InvalidParameterError(
                f"X and Y must have the same number of rows. Got {X.shape} and {Y.shape}."
            )
        if Y.shape[1] != self.n_objectives:
            raise InvalidParameterError(
                f"Y must have n_objectives={self.n_objectives} columns. Got {Y.shape}."
            )
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
            raise InvalidParameterError("X and Y must contain only finite values.")
        lower = np.asarray(self.LowerBounds, dtype=float).reshape(1, -1)
        upper = np.asarray(self.UpperBounds, dtype=float).reshape(1, -1)
        if np.any(X < lower) or np.any(X > upper):
            raise InvalidParameterError("X contains values outside the declared parameter bounds.")

        if failed is None:
            failed_mask = np.zeros(Y.shape[0], dtype=bool)
        else:
            failed_mask = np.asarray(failed).reshape(-1).astype(bool)
            if failed_mask.shape != (Y.shape[0],):
                raise InvalidParameterError(
                    f"failed must be a ({Y.shape[0]},) boolean mask, one entry per "
                    f"observation. Got shape {np.shape(failed)}."
                )

        supplied_yvar = Yvar is not None
        if Yvar is not None:
            Yvar = np.atleast_2d(np.asarray(Yvar, dtype=float))
            if Yvar.shape != Y.shape:
                raise InvalidParameterError(
                    f"Yvar shape {Yvar.shape} must match Y shape {Y.shape}."
                )
            if not np.all(np.isfinite(Yvar)) or np.any(Yvar < 0):
                self._fall_back_to_inferred_noise(
                    "Non-finite or negative values in Yvar; falling back to "
                    "inferred noise for the whole run."
                )
                Yvar = None

        new_train_Yvar: np.ndarray | None
        if self.train_X is None:
            if (
                Yvar is None
                and self._constructor_train_Yvar is not None
                and not self._mc_uncertainty_fallback
            ):
                Yvar = self._use_constructor_train_Yvar(Y)
            new_train_X = X.copy()
            new_train_Y = Y.copy()
            new_train_failed = failed_mask.copy()
            new_train_Yvar = None if self._mc_uncertainty_fallback else Yvar
        else:
            if self.train_Y is None:
                raise InvalidParameterError("Stored train_X/train_Y state is desynchronized.")
            new_train_X = np.vstack([self.train_X, X])
            new_train_Y = np.vstack([self.train_Y, Y])
            new_train_failed = np.concatenate([self._failed_prefix(len(self.train_Y)), failed_mask])
            if self._mc_uncertainty_fallback:
                if supplied_yvar and Yvar is not None:
                    warnings.warn(
                        "Yvar was supplied after this run had already fallen "
                        "back to inferred noise; ignoring it.",
                        UserWarning,
                        stacklevel=2,
                    )
                new_train_Yvar = None
            elif Yvar is None:
                if self.train_Yvar is not None:
                    self._fall_back_to_inferred_noise(
                        "Observations were told without Yvar after earlier "
                        "observations supplied it; falling back to inferred "
                        "noise for the whole run."
                    )
                new_train_Yvar = None
            elif self.train_Yvar is None:
                self._fall_back_to_inferred_noise(
                    "Yvar was supplied after earlier observations used inferred "
                    "noise; falling back to inferred noise for the whole run."
                )
                new_train_Yvar = None
            else:
                new_train_Yvar = np.vstack([self.train_Yvar, Yvar])

        # Run the user constraint callables over the incoming rows before
        # committing anything. Earlier rows were validated when they were
        # committed, so only the new ones are scored -- and the answer seeds the
        # feasibility cache rather than being recomputed on the next read.
        prior_n = 0 if self.train_Y is None else len(self.train_Y)
        prior_feasible = np.zeros(0, dtype=bool) if prior_n == 0 else self._feasible_cache
        new_rows_feasible = self._observed_feasible_mask(X, Y)

        self.train_X = new_train_X
        self.train_Y = new_train_Y
        self.train_failed = new_train_failed
        self.train_Yvar = new_train_Yvar
        self._feasible_cache = (
            np.concatenate([prior_feasible, new_rows_feasible])
            if prior_feasible is not None and len(prior_feasible) == prior_n
            else None  # unknown prefix; feasible_mask() rebuilds it on demand
        )

        self.PopulationHistory.append((self._batch_index, Y.copy()))
        # PopulationHistory stores (batch_index, batch_Y) — the observations
        # just told — matching the Phase-1 plotting contract. NSGA-II stores
        # (generation_index, population_F) instead; both are
        # (step_index, objectives_array) snapshots consumed by the same plotter.
        self._update_pareto_attrs()
        hv = self._update_hv_history()
        self._batch_index += 1
        self._pending_X = None

        try:
            self.save_checkpoint()
        except (OSError, ValueError) as exc:
            logger.warning("Could not write MOBO checkpoint: %s", exc)

        logger.info(
            "tell: n_obs=%s n_failed=%s hypervolume=%s n_pareto=%s",
            len(self.train_X),
            int(self.train_failed.sum()),
            hv,
            len(self.ParetoObjectives),
        )

    def _objective_variance_callable(self) -> Callable | None:
        """Resolve the user's optional ``TopasObjectiveVariances`` once per run.

        The lookup is memoized because it executes the user's objective module:
        re-importing it per evaluation would repeat any import-time side effects
        that file has, once for every candidate in the campaign.
        """
        if self._variance_fn is _UNRESOLVED:
            from .utilities import _import_from_absolute_path

            obj_path = Path(self.OptimizationDirectory) / "TopasObjectiveFunction.py"
            try:
                objective_mod = _import_from_absolute_path(obj_path)
            except ModuleNotFoundError:
                self._variance_fn = None
            else:
                var_fn = getattr(objective_mod, "TopasObjectiveVariances", None)
                self._variance_fn = var_fn if callable(var_fn) else None
        return self._variance_fn

    def _try_extract_mc_variances(
        self,
        iteration: int,
        objective_values: np.ndarray,
    ) -> np.ndarray | None:
        """Load optional per-objective variances for a specific evaluation.

        Looks for ``TopasObjectiveVariances(ResultsLocation, iteration)`` in
        ``TopasObjectiveFunction.py``. Returns ``None`` if absent. Variances
        must already reflect any arithmetic used to form the objectives. User
        objective modules can call `propagate_objective_variance` with
        scorer means and an analytic Jacobian for ratios or other transforms.

        :param iteration: The ``evaluation_index`` that actually produced the
            results being scored. This must be captured *before* calling
            ``EvaluateObjectives``, which advances the counter.
        """
        var_fn = self._objective_variance_callable()
        if var_fn is None:
            return None
        results_location = str(Path(self.BaseDirectory) / self.SimulationName / "Results")
        try:
            raw = var_fn(results_location, int(iteration))
            var = propagate_objective_variance(
                np.asarray(objective_values, dtype=float),
                raw,
            )
        except Exception as exc:
            logger.warning(
                "TopasObjectiveVariances failed (%s); will fall back if required.",
                exc,
            )
            return None
        if var.shape != (self.n_objectives,):
            logger.warning(
                "TopasObjectiveVariances returned %d values but n_objectives=%d; "
                "will fall back if required.",
                var.size,
                self.n_objectives,
            )
            return None
        return var

    def _evaluate_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Evaluate candidates; return ``(Y, Yvar_or_None, failed)``.

        ``Y`` is in minimize space; ``failed`` is a ``(n,)`` boolean mask
        marking rows where the base class substituted ``failure_penalty``
        instead of a real measurement.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self._objective_fn is not None:
            Y = np.atleast_2d(np.asarray(self._objective_fn(X), dtype=float))
            # Synthetic evaluators are expected to raise rather than penalize.
            return Y, None, np.zeros(Y.shape[0], dtype=bool)

        rows = []
        yvars = []
        failed_rows = []
        use_var = self.use_mc_uncertainty and not self._mc_uncertainty_fallback
        for row in X:
            # Capture the iteration and cache state *before* EvaluateObjectives:
            # it advances evaluation_index even when it serves a cached design
            # without re-running TOPAS, so reading variances for
            # ``evaluation_index - 1`` afterwards can point at an iteration whose
            # Results were never produced (or belong to a different design).
            iteration = self.evaluation_index
            cache_key = self._cache_key(row)
            was_cached = cache_key in self._eval_cache
            y = np.asarray(self.EvaluateObjectives(row), dtype=float).reshape(-1)
            rows.append(y[: self.n_objectives])
            # The base class records this for every evaluation, including ones
            # replayed from the cache on a resumed run, so a design whose real
            # objectives happen to equal failure_penalty is not misread as a
            # crash (and vice versa).
            row_failed = self._last_evaluation_failed
            failed_rows.append(row_failed)
            if use_var:
                if row_failed:
                    # No Results were produced for a failed evaluation, and
                    # _fit_model excludes the row anyway. Keep a placeholder so
                    # train_Yvar stays row-aligned with train_Y rather than
                    # collapsing the whole run to inferred noise over a crash.
                    yvars.append(np.zeros(self.n_objectives, dtype=float))
                    continue
                if was_cached:
                    # Reuse the variance recorded when this design was actually
                    # simulated; its Results directory may be long gone.
                    var = self._variance_cache.get(cache_key)
                else:
                    var = self._try_extract_mc_variances(
                        iteration,
                        y[: self.n_objectives],
                    )
                    if var is not None:
                        self._variance_cache[cache_key] = var
                if var is None or not np.all(np.isfinite(var)):
                    self._fall_back_to_inferred_noise(
                        "use_mc_uncertainty=True but objective variances were "
                        f"missing or non-finite for iteration {iteration}; "
                        "falling back to inferred noise for the whole run."
                    )
                    use_var = False
                    yvars = []
                else:
                    yvars.append(np.asarray(var, dtype=float).reshape(-1))
        Y = np.vstack(rows)
        Yvar = np.vstack(yvars) if use_var and yvars else None
        failed = np.asarray(failed_rows, dtype=bool)
        if failed.any():
            logger.warning(
                "%d of %d evaluations in this batch were penalized failures; "
                "they are excluded from GP training, the reference points, and "
                "the reported Pareto front.",
                int(failed.sum()),
                len(failed),
            )
        return Y, Yvar, failed

    def run(self, n_batches: int | None = None) -> Any:
        """Closed-loop MOBO: initial design + ``n_batches`` acquisition steps.

        :param n_batches: Acquisition batches after initialization. Defaults to
            ``optimization_params['n_generations']`` / ``n_iterations`` (the
            shared algorithm-step key used by ``NSGAII_Optimizer`` for
            generations).
        :returns: Simple result object with ``.X`` / ``.F`` (Pareto set),
            matching the attribute names plotting reads from
            ``NSGAII_Optimizer``.
        """
        n_batches = self._n_batches_target if n_batches is None else int(n_batches)
        if n_batches < 0:
            raise InvalidParameterError(f"n_batches must be >= 0. Got {n_batches}.")

        # Also fires the resume hook (_restore_algorithm_state -> load_checkpoint)
        # when resume=True. No-op when the caller already set the directories up.
        self.SetUpDirectoryStructure()

        if self.train_X is None or len(self.train_X) == 0:
            X0 = self.ask()
            Y0, V0, F0 = self._evaluate_batch(X0)
            self.tell(X0, Y0, V0, failed=F0)

        # After the Sobol tell, _batch_index == 1. Run n_batches acquisition rounds.
        while max(0, self._batch_index - 1) < n_batches:
            Xb = self.ask()
            Yb, Vb, Fb = self._evaluate_batch(Xb)
            self.tell(Xb, Yb, Vb, failed=Fb)

        self._finalize_results()
        return self.res

    def _finalize_results(self) -> None:
        """Write the official final front and run artifacts (NSGA-II parity).

        Official final front: eligible non-dominated observations
        (``ParetoObjectives`` / ``ParetoDecisionVars`` → ``res.F`` / ``res.X``),
        written to ``ParetoFront.txt``. Mid-run monitoring uses
        ``ParetoFront_Running.txt`` (updated in ``_update_pareto_attrs``).
        """
        self._update_pareto_attrs()
        # Ensure HV history exists
        if not self.HypervolumeHistory and self.train_Y is not None:
            self._update_hv_history()

        self.res = SimpleNamespace(
            F=np.asarray(self.ParetoObjectives, dtype=float),
            X=np.asarray(self.ParetoDecisionVars, dtype=float),
        )

        LogParetoFrontToFile(
            self._ParetoLogFileLoc,
            self.ParetoObjectives,
            self.ParameterNames,
            self.n_objectives,
            ParetoDecisionVars=self.ParetoDecisionVars,
        )
        self._write_final_log_entry()
        self._persist_run_state()

        try:
            self.save_checkpoint()
        except (OSError, ValueError) as exc:
            logger.warning("Could not write final MOBO checkpoint: %s", exc)

    def RunOptimization(self):
        """Set up directories and run the closed-loop MOBO campaign (TOPAS path).

        Uses ``EvaluateObjectives`` for every candidate unless ``objective_fn``
        was supplied at construction (synthetic/benchmark mode). Mirrors
        ``NSGAII_Optimizer.RunOptimization``: resume via the base-class
        directory setup, write ``ParetoFront.txt`` / logs at the end, then
        produce intermediate-style and final visualizations.

        :returns: Result object with ``.F`` / ``.X`` Pareto arrays (also stored
            as ``self.res``), matching the plotting contract used by NSGA-II.
        """
        self.SetUpDirectoryStructure()
        # The directories now exist, so re-derive the checkpoint path through the
        # single place that owns it rather than rebuilding it here.
        self._mobo_checkpoint_loc = None
        self._mobo_ckpt_path()
        logger.info(
            "Starting MOBO (%s) n_init=%s batch_size=%s n_batches=%s d=%s m=%s",
            self._acquisition_resolved,
            self.n_init,
            self.batch_size,
            self._n_batches_target,
            len(self.ParameterNames),
            self.n_objectives,
        )
        result = self.run()
        self._plot_convergence()
        self.GenerateFinalVisualizations()
        return result
