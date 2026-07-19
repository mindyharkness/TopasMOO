# -*- coding: utf-8 -*-
"""
Multi-objective optimization drivers for TopasMOO (pymoo integration).

Extends the TopasOpt workflow to multiple objectives and Pareto-front tracking.
"""
import json
import logging
import numbers
import os
import pickle
import shlex
import shutil
import stat
import subprocess
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination

from .exceptions import (
    InvalidParameterError,
    MalformedOutputError,
    ObjectiveFunctionError,
    TopasExecutionError,
)
from .io import LogParetoFrontToFile, ReadInMultiObjectiveLogFile
from .metrics import calculate_dominance_rank
from .plotting import GenerateComprehensiveVisualizations
from .plotting.convergence import plot_objective_convergence, plot_parameter_convergence
from .plotting.pareto import plot_pareto_front
from .plotting.style import (
    INTERMEDIATE_PLOT_DPI,
    apply_style,
    available_publication_variants,
)
from .utilities import _import_from_absolute_path, _load_user_callable

logger = logging.getLogger(__name__)


class TopasMOOBaseClass(ABC):
    """Base class for multi-objective TOPAS Monte Carlo optimization.

    Provides logging, objective evaluation, script generation, Pareto tracking,
    and shared directory setup for pymoo-backed optimizers.

    Compared to TopasOpt:
        - Vector-valued objectives instead of a single fitness.
        - Pareto front tracking instead of one best solution.
        - pymoo as the optimization driver.
        - Multi-objective plotting helpers.

    Do not instantiate this class directly; subclass it (e.g. ``NSGAII_Optimizer``).
    Constructor parameters are documented on ``__init__``.
    """

    def __init__(
        self,
        optimization_params,
        BaseDirectory,
        SimulationName,
        OptimizationDirectory,
        ReadMeText=None,
        G4dataLocation="~/G4Data",
        TopasLocation="~/topas/",
        ShellScriptHeader=None,
        Overwrite=False,
        KeepAllResults=True,
        plot_frequency=10,
        final_plots=None,
        plot_style="publication",
        intermediate_plot_style="fast",
        publication_variant="clean",
        n_constraints=0,
        on_evaluation_failure="penalize",
        failure_penalty=1e6,
        resume=False,
        dump_optimization_settings=False,
    ):
        """Initialize shared TOPAS multi-objective state (subclasses only).

        :param optimization_params: Dict including ``n_objectives`` (>= 2), ``ParameterNames``,
            ``start_point``, ``UpperBounds``, ``LowerBounds``, and ``n_generations``
            (the number of NSGA-II generations to run; the legacy alias
            ``n_iterations`` is still accepted). ``start_point`` is injected
            into the initial NSGA-II population as one individual (the rest
            are sampled randomly); it is ignored when a ``custom_algorithm``
            supplies its own sampling or when resuming from a checkpoint.
        :param BaseDirectory: Existing root directory for simulation outputs.
        :param SimulationName: Subfolder name for this optimization run.
        :param OptimizationDirectory: Directory containing ``GenerateTopasScripts.py`` and
            ``TopasObjectiveFunction.py``.
        :param ReadMeText: If set, written to ``readme.txt`` under the simulation folder.
        :param G4dataLocation: Geant4 data directory (may use ``~``).
        :param TopasLocation: TOPAS install root, or the string ``'testing_mode'`` for tests.
        :param ShellScriptHeader: Optional bash preamble for ``RunIteration.sh``.
        :param Overwrite: If True, clear a non-empty simulation folder without prompting.
        :param KeepAllResults: If False, clear ``Results`` before each evaluation.
        :param plot_frequency: Number of objective *evaluations* between intermediate
            convergence plots (not generations); final plots always run at the end.
            Must be a positive integer.
        :param final_plots: End-of-run plot selection: ``None`` / ``"default"``
            for the lean default set
            (:data:`~TopasMOO.plotting.DEFAULT_FINAL_PLOTS`), ``"all"`` for
            every recognized key, a single key string (e.g. ``"pareto"``), or
            an iterable of keys. See
            :func:`~TopasMOO.plotting.GenerateComprehensiveVisualizations` for all
            recognized keys.
        :param plot_style: Final-figure style for end-of-run visualizations.
            One of ``"fast"`` or ``"publication"`` (default).
        :param intermediate_plot_style: Style for iterative plots generated during
            optimization. One of ``"fast"`` (default) or ``"publication"``.
        :param publication_variant: Variant of the ``"publication"`` style. One of
            ``"clean"`` (default), ``"nature"``, ``"ieee"``, or
            ``"medicalphysics"`` -- see :mod:`TopasMOO.plotting.style` for what
            each looks like.
            Ignored when both styles are ``"fast"``.
        :param n_constraints: Number of inequality constraints (``g(x) <= 0`` is
            feasible, pymoo convention). When ``> 0`` the user
            ``TopasObjectiveFunction`` must return ``n_objectives +
            n_constraints`` values: the objectives first, then the
            constraint values. Default ``0`` (unconstrained).
        :param on_evaluation_failure: ``"penalize"`` (default) or ``"raise"``.
            With ``"penalize"``, a TOPAS run that exits non-zero, an
            objective function that raises, or a non-finite objective value
            is logged and assigned ``failure_penalty`` (and an infeasible
            constraint value) so the optimization continues instead of
            aborting. Contract violations (wrong type/shape/length of the
            returned vector) always raise regardless of this setting.
        :param failure_penalty: Objective value assigned to each objective of a
            failed evaluation when ``on_evaluation_failure="penalize"``.
            Must be a finite, positive number, and worse (larger, for
            minimization) than any real objective so failed designs are
            dominated. Default ``1e6``.
        :param resume: If True, continue a previous run in the same simulation
            folder: the evaluation cache, ``evaluation_index``
            (``logs/RunState.json``), and (if present) the per-generation
            algorithm checkpoint are loaded so completed simulations are not
            repeated and iteration numbering continues. The simulation folder
            is not cleared when resuming.
        :param dump_optimization_settings: If True, write a jsonpickle snapshot
            of constructor state to ``OptimizationSettings.json`` when the
            simulation folder is first created. Default ``False`` (not used by
            resume; enable only for debugging or archival). Requires the
            optional ``jsonpickle`` dependency:
            ``pip install TopasMOO[settings-dump]``.

        :raises TypeError: If ``TopasMOOBaseClass`` is constructed directly.
        :raises InvalidParameterError: If ``n_objectives``, ``n_constraints``,
            ``on_evaluation_failure``, or bounds/start point are invalid.
        :raises FileNotFoundError: If ``BaseDirectory`` or the TOPAS binary is missing.
        :raises ModuleNotFoundError: If required user scripts cannot be imported.
        """

        # ``TopasMOOBaseClass`` is an ABC with an abstract ``RunOptimization``,
        # so attempting to instantiate it directly already raises TypeError.

        # Work on a copy so we never mutate the caller's dict.
        optimization_params = dict(optimization_params)
        optimization_params = self._normalize_iteration_key(optimization_params)

        if optimization_params.get("start_point") is None:
            msg = (
                "optimization_params must include a non-None 'start_point' "
            )
            logger.error(msg)
            raise InvalidParameterError(msg)

        optimization_params = self._convert_optimization_params_to_numpy(
            optimization_params
        )

        # Check for multi-objective requirement
        if "n_objectives" not in optimization_params:
            msg = (
                'TopasMOO requires "n_objectives" key in optimization_params. '
            )
            logger.error(msg)
            raise InvalidParameterError(msg)

        self.n_objectives = int(optimization_params["n_objectives"])
        if self.n_objectives < 2:
            msg = (
                f"n_objectives must be >= 2 for multi-objective optimization. "
                f"You specified {self.n_objectives}. "
                f"Use TopasOpt for single-objective problems."
            )
            logger.error(msg)
            raise InvalidParameterError(msg)

        self.n_constraints = int(n_constraints)
        if self.n_constraints < 0:
            raise InvalidParameterError(
                f"n_constraints must be >= 0. Got {self.n_constraints}."
            )
        if on_evaluation_failure not in {"penalize", "raise"}:
            raise InvalidParameterError(
                "on_evaluation_failure must be 'penalize' or 'raise'. "
                f"Got '{on_evaluation_failure}'."
            )
        self.on_evaluation_failure = on_evaluation_failure
        self.failure_penalty = float(failure_penalty)
        # A penalty must be a finite, strictly positive number: non-finite values
        # break pymoo's dominance/hypervolume math, and 0 (or negative) would make
        # a failed evaluation look as good as (or better than) a real one.
        if not np.isfinite(self.failure_penalty) or self.failure_penalty <= 0:
            raise InvalidParameterError(
                "failure_penalty must be a finite, positive number. "
                f"Got {failure_penalty!r}."
            )
        self.resume = bool(resume)
        self.dump_optimization_settings = bool(dump_optimization_settings)
        self._n_failed_evaluations = 0
        self._last_constraint_values = None
        self._eval_cache = {}

        # Expand ``~`` here (the export line in RunIteration.sh is shell-quoted,
        # which would otherwise prevent bash from expanding a literal ``~``).
        self.G4dataLocation = os.path.expanduser(str(G4dataLocation))
        self.ReadMeText = ReadMeText
        self.ShellScriptHeader = ShellScriptHeader
        self.KeepAllResults = KeepAllResults
        self.BaseDirectory = BaseDirectory
        self.OptimizationDirectory = OptimizationDirectory

        if not os.path.isdir(BaseDirectory):
            msg = f'Input BaseDirectory "{BaseDirectory}" does not exist.'
            logger.error(msg)
            raise FileNotFoundError(msg)

        self.SimulationName = SimulationName
        _LogFileLoc = Path(self.BaseDirectory) / self.SimulationName
        _LogFileLoc = _LogFileLoc / "logs"
        self._LogFileLoc = str(_LogFileLoc / "OptimizationLogs.txt")
        self._ParetoLogFileLoc = str(_LogFileLoc / "ParetoFront.txt")
        self._ParetoRunningLogFileLoc = str(_LogFileLoc / "ParetoFront_Running.txt")
        self._eval_cache_loc = str(_LogFileLoc / "EvalCache.jsonl")
        self._checkpoint_loc = str(_LogFileLoc / "Checkpoint.pkl")
        self._run_state_loc = str(_LogFileLoc / "RunState.json")

        self.evaluation_index = 0
        self._optimization_params = optimization_params

        # Set up optimization parameters
        self.ParameterNames = optimization_params["ParameterNames"]

        # Canonical start-point shape: a 1-D float vector of length n_params.
        self.StartingValues = np.asarray(
            optimization_params["start_point"], dtype=float
        ).reshape(-1)
        self.x = self.StartingValues

        self.UpperBounds = optimization_params["UpperBounds"]
        self.LowerBounds = optimization_params["LowerBounds"]
        self.n_generations = int(optimization_params["n_iterations"])
        self._create_variable_dictionary(self.StartingValues)
        self.Overwrite = Overwrite

        # Plot configuration. Reject bools and non-integral numbers outright
        # rather than coercing: int(2.9) == 2 would silently plot at a
        # different cadence than the caller asked for.
        if isinstance(plot_frequency, bool) or not isinstance(
            plot_frequency, numbers.Integral
        ):
            raise InvalidParameterError(
                "plot_frequency must be a positive integer "
                f"(number of evaluations between intermediate plots). "
                f"Got {plot_frequency!r}."
            )
        self.plot_frequency = int(plot_frequency)
        if self.plot_frequency < 1:
            raise InvalidParameterError(
                "plot_frequency must be a positive integer "
                f"(number of evaluations between intermediate plots). "
                f"Got {plot_frequency!r}."
            )
        self.final_plots = final_plots
        self.plot_style = plot_style
        self.intermediate_plot_style = intermediate_plot_style
        self.publication_variant = publication_variant
        valid_plot_styles = {"fast", "publication"}
        valid_publication_variants = set(available_publication_variants())
        if self.plot_style not in valid_plot_styles:
            raise InvalidParameterError(
                f"plot_style must be one of {sorted(valid_plot_styles)}. "
                f"Got '{self.plot_style}'."
            )
        if self.intermediate_plot_style not in valid_plot_styles:
            raise InvalidParameterError(
                f"intermediate_plot_style must be one of {sorted(valid_plot_styles)}. "
                f"Got '{self.intermediate_plot_style}'."
            )
        if self.publication_variant not in valid_publication_variants:
            raise InvalidParameterError(
                f"publication_variant must be one of "
                f"{sorted(valid_publication_variants)}. "
                f"Got '{self.publication_variant}'."
            )

        # Multi-objective specific tracking
        self.AllObjectiveFunctionValues = []  # List of arrays, one array per iteration
        self.AllDecisionVariables = []        # Parallel list of decision-variable vectors
        self.ParetoObjectives = []            # Objective values of the non-dominated set
        self.ParetoDecisionVars = None        # Decision variables for Pareto set
        self.HypervolumeHistory = []          # Track hypervolume per generation
        self.PopulationHistory = []           # Track population objectives per generation

        if "~" in str(TopasLocation):
            TopasLocation = os.path.expanduser(str(TopasLocation))
        self.TopasLocation = Path(TopasLocation)
        self._testing_mode = False
        if str(self.TopasLocation) == "testing_mode":
            self._testing_mode = True
            logger.warning(
                f"Entering test mode because topas location = {self.TopasLocation}"
            )

        # Load user defined model generator and objective function
        try:
            generate_mod = _import_from_absolute_path(
                Path(self.OptimizationDirectory) / "GenerateTopasScripts.py"
            )
        except ModuleNotFoundError as e:
            logger.error(
                f'Failed to import required file at {str(Path(self.OptimizationDirectory) / "GenerateTopasScripts.py")}.'
                f"\nQuitting"
            )
            raise e
        try:
            objective_mod = _import_from_absolute_path(
                Path(self.OptimizationDirectory) / "TopasObjectiveFunction.py"
            )
        except ModuleNotFoundError as e:
            logger.error(
                f'Failed to import required file at {str(Path(self.OptimizationDirectory) / "TopasObjectiveFunction.py")}.'
                f"\nQuitting"
            )
            raise e
        self.TopasScriptGenerator = _load_user_callable(
            generate_mod,
            "GenerateTopasScripts",
            Path(self.OptimizationDirectory) / "GenerateTopasScripts.py",
            "GenerateTopasScripts(BaseDirectory, iteration, **parameters)",
        )
        self.TopasObjectiveFunction = _load_user_callable(
            objective_mod,
            "TopasObjectiveFunction",
            Path(self.OptimizationDirectory) / "TopasObjectiveFunction.py",
            "TopasObjectiveFunction(ResultsLocation, iteration)",
        )
        self._check_input_data()

    @staticmethod
    def _normalize_iteration_key(optimization_params):
        """Normalize the generation-count key to the internal ``n_iterations``.

        The number of NSGA-II *generations* to run may be supplied as:

        * ``n_generations`` — preferred public spelling;
        * ``n_iterations`` — accepted alias kept for backwards compatibility;
        * ``Nitterations`` — deprecated legacy spelling (emits ``FutureWarning``).

        A "generation" is one NSGA-II population step; it is **not** the same as
        the per-evaluation ``evaluation_index`` counter (see
        ``EvaluateObjectives``) or ``plot_frequency``, both of which count
        individual objective evaluations.

        :param optimization_params: Parameter dict; may be updated in place.

        :returns: The same dict with a single ``n_iterations`` entry.

        :raises ValueError: If no generation-count key is present.
        """
        if "Nitterations" in optimization_params:
            warnings.warn(
                "The 'Nitterations' parameter key is deprecated and will be removed "
                "in a future release. Use 'n_generations' instead.",
                FutureWarning,
                stacklevel=3,
            )
            optimization_params.setdefault(
                "n_generations", optimization_params.pop("Nitterations")
            )

        # 'n_generations' is the canonical public name; fall back to the
        # 'n_iterations' alias. Internally we always store 'n_iterations'.
        if "n_generations" in optimization_params:
            if (
                "n_iterations" in optimization_params
                and optimization_params["n_iterations"] != optimization_params["n_generations"]
            ):
                warnings.warn(
                    "Both 'n_generations' and 'n_iterations' supplied with different values; using 'n_generations'.",
                    FutureWarning,
                    stacklevel=3,
                )
            optimization_params["n_iterations"] = optimization_params.pop("n_generations")

        if "n_iterations" not in optimization_params:
            raise ValueError(
                "optimization_params must contain 'n_generations' "
                "(number of NSGA-II generations to run)."
            )
        return optimization_params

    def _convert_optimization_params_to_numpy(self, optimization_params):
        """Coerce list-valued optimization entries to float ``numpy`` arrays.

        :param optimization_params: Parameter dict; updated in place.

        :returns: ``optimization_params`` after coercion.

        :raises InvalidParameterError: If an array-valued entry is neither a list
            nor a ``numpy`` array (and so cannot be coerced).
        """
        skip_keys = {
            "ParameterNames",
            "Nitterations",
            "n_iterations",
            "n_generations",
            "n_objectives",
        }
        for param_key in list(optimization_params.keys()):
            if param_key in skip_keys:
                continue
            value = optimization_params[param_key]
            if isinstance(value, list):
                value = np.array(value)
            if not isinstance(value, np.ndarray):
                msg = (
                    f"optimization param '{param_key}' must be a list or numpy "
                    f"array, got {type(value).__name__}."
                )
                logger.error(msg)
                raise InvalidParameterError(msg)
            optimization_params[param_key] = value.astype(float)

        return optimization_params

    def _create_variable_dictionary(self, x):
        """Map ``ParameterNames`` to scalar values in ``self.VariableDict``.

        :param x: 1D vector or single-row 2D array of parameters.

        :raises InvalidParameterError: If ``x`` is not 1D or 2D in the expected layout.
        """
        if np.ndim(x) == 1:
            self.VariableDict = {
                self.ParameterNames[i]: x[i] for i in range(len(self.ParameterNames))
            }
        elif np.ndim(x) == 2:
            self.VariableDict = {
                self.ParameterNames[i]: x[0][i] for i in range(len(self.ParameterNames))
            }
        else:
            msg = (
                f"Parameter vector must be 1D or a single-row 2D array; "
                f"got {np.ndim(x)} dimensions."
            )
            logger.error(msg)
            raise InvalidParameterError(msg)

        for key in self.VariableDict.keys():
            if isinstance(self.VariableDict[key], np.ndarray):
                self.VariableDict[key] = self.VariableDict[key][0]

    def _empty_simulation_folder(self):
        """Clear the simulation folder if it has contents and ``Overwrite`` is True.

        :raises RuntimeError: If the directory is not empty and ``Overwrite`` is
            False. Set ``Overwrite=True`` to opt in to clearing.
        """
        SimName = str(Path(self.BaseDirectory) / self.SimulationName)
        if not bool(os.listdir(SimName)):
            return

        if not self.Overwrite:
            raise RuntimeError(
                f"Simulation folder '{SimName}' is not empty. "
                "Pass Overwrite=True to clear it, or use a new SimulationName."
            )

        logger.warning("Emptying simulation folder %s", SimName)
        for filename in os.listdir(SimName):
            file_path = os.path.join(SimName, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error("Failed to delete %s.", file_path)
                raise e

    def _check_input_data(self):
        """Validate dimensions, bounds, starting point, and TOPAS binary (unless testing).

        :raises InvalidParameterError: If sizes or bound violations occur.
        :raises FileNotFoundError: If the TOPAS executable is missing in normal mode.
        """
        if not np.size(self.ParameterNames) == np.size(self.StartingValues):
            msg = "size of ParameterNames does not match size of StartingValues"
            logger.error(msg)
            raise InvalidParameterError(msg)
        if not np.size(self.StartingValues) == np.size(self.UpperBounds):
            msg = "size of StartingValues does not match size of UpperBounds"
            logger.error(msg)
            raise InvalidParameterError(msg)
        if not np.size(self.UpperBounds) == np.size(self.LowerBounds):
            msg = "size of UpperBounds does not match size of LowerBounds"
            logger.error(msg)
            raise InvalidParameterError(msg)

        for i, Parameter in enumerate(self.ParameterNames):
            start_value = self.StartingValues[i]
            if start_value < self.LowerBounds[i]:
                msg = (
                    f"For {Parameter}, Starting value {start_value} "
                    f"is less than Lower bound {self.LowerBounds[i]}"
                )
                logger.error(msg)
                raise InvalidParameterError(msg)
            elif start_value > self.UpperBounds[i]:
                msg = (
                    f"For {Parameter}, Starting value {start_value} "
                    f"is greater than upper bound {self.UpperBounds[i]}"
                )
                logger.error(msg)
                raise InvalidParameterError(msg)

        if not self._testing_mode:
            if not os.path.isfile(self.TopasLocation / "bin" / "topas"):
                error = (
                    f"could not find topas binary at \n{self.TopasLocation} "
                    f"\nPlease initialize with TopasLocation pointing to the topas installation location."
                    f"\nQuitting"
                )
                raise FileNotFoundError(error)

    def _generate_topas_model(self):
        """Write TOPAS scripts for the current iteration and refresh ``RunIteration.sh``.

        Reads the current design from ``self.VariableDict`` (set by
        ``_create_variable_dictionary`` just before this is called).
        """
        self.TopasScripts, self.TopasScriptNames = self.TopasScriptGenerator(
            Path(self.BaseDirectory) / self.SimulationName,
            self.evaluation_index,
            **self.VariableDict,
        )

        self.ScriptsToRun = []
        for i, script_name in enumerate(self.TopasScriptNames):
            script_name = script_name + "_itt_" + str(self.evaluation_index) + ".tps"
            self.ScriptsToRun.append(script_name)
            script_path = (
                Path(self.BaseDirectory)
                / self.SimulationName
                / "TopasScripts"
                / script_name
            )
            with open(str(script_path), "w") as f:
                for line in self.TopasScripts[i]:
                    f.write(line)
                    f.write("\n")

        self._generate_run_iteration_shell_script()

    def _setup_topas_emulator(self):
        """Install a minimal executable named ``topas`` that only echoes, for tests."""
        if not os.path.isdir(Path(self.BaseDirectory) / self.SimulationName / "bin"):
            os.mkdir(Path(self.BaseDirectory) / self.SimulationName / "bin")

        EmulatorLocation = (
            Path(self.BaseDirectory) / self.SimulationName / "bin" / "topas"
        )
        if os.path.isfile(EmulatorLocation):
            os.remove(EmulatorLocation)
        self.TopasLocation = EmulatorLocation.parent.parent

        with open(EmulatorLocation, "w") as f:
            f.write(
                'echo "Hello from topas emulator! I dont do anything except print this"'
            )
        st = os.stat(EmulatorLocation)
        os.chmod(EmulatorLocation, st.st_mode | stat.S_IEXEC)

    def _generate_run_iteration_shell_script(self):
        """Write executable ``RunIteration.sh`` under ``TopasScripts`` for this iteration."""
        ShellScriptLocation = str(
            Path(self.BaseDirectory)
            / self.SimulationName
            / "TopasScripts"
            / "RunIteration.sh"
        )
        if os.path.isfile(ShellScriptLocation):
            os.remove(ShellScriptLocation)
        self.ShellScriptLocation = ShellScriptLocation

        topas_bin = shlex.quote(str(Path(self.TopasLocation) / "bin" / "topas"))

        with open(ShellScriptLocation, "w") as f:
            # Always establish the TOPAS environment (shebang + Geant4 data dir).
            # A custom ShellScriptHeader is appended *after* this so it augments
            # the environment (e.g. cluster module loads) rather than silently
            # replacing the TOPAS_G4_DATA_DIR export.
            f.write("#!/bin/bash\n")
            f.write("# This script sets up the topas environment then runs all listed files\n")
            f.write(
                f"export TOPAS_G4_DATA_DIR={shlex.quote(str(self.G4dataLocation))}\n"
            )
            if self.ShellScriptHeader is not None:
                f.write(self.ShellScriptHeader)
                f.write("\n")

            for script_name in self.ScriptsToRun:
                quoted_script = shlex.quote(script_name)
                quoted_log = shlex.quote(f"../logs/TopasLogs/{script_name}")
                f.write(f'echo "Beginning analysis of: {script_name}"\n')
                f.write(
                    f"(time TOPAS_HEADLESS_MODE=1 {topas_bin} {quoted_script}) "
                    f"&> {quoted_log}\n"
                )

        st = os.stat(ShellScriptLocation)
        os.chmod(ShellScriptLocation, st.st_mode | stat.S_IEXEC)

    def _run_topas_model(self):
        """Run ``RunIteration.sh`` via bash from the ``TopasScripts`` directory.

        :raises TopasExecutionError: If the shell script exits with a non-zero status.
        """
        logger.info("Topas: Running file: %s", self.ShellScriptLocation)
        ShellScriptPath = str(
            Path(self.BaseDirectory) / self.SimulationName / "TopasScripts"
        )
        cmd = subprocess.run(["bash", self.ShellScriptLocation], cwd=ShellScriptPath)
        if cmd.returncode == 0:
            logger.info("Analysis complete")
        else:
            topas_log_dir = (
                Path(self.BaseDirectory) / self.SimulationName / "logs" / "TopasLogs"
            )
            logger.error(
                f"RunIteration.sh failed with exit code {cmd.returncode}."
                f"\nSuggestion: look at {topas_log_dir} "
                f"\nto figure out what went wrong... Quitting"
            )
            raise TopasExecutionError(
                f"RunIteration.sh failed with exit code {cmd.returncode}."
                f"\nSuggestion: look at {topas_log_dir} "
                f"\nto figure out what went wrong... Quitting"
            )

    def _update_optimization_logs(self, x, objective_values):
        """Append iteration, parameters, and objectives to the optimization log file.

        :param x: Current parameters (1D or single-row 2D).
        :param objective_values: Objective vector for this evaluation.
        """
        with open(self._LogFileLoc, "a") as f:
            Entry = f"Iteration: {self.evaluation_index}"
            for i, Parameter in enumerate(self.ParameterNames):
                try:
                    Entry = Entry + f", {Parameter}: {x[0][i]: 1.2f}"
                except IndexError:
                    Entry = Entry + f", {Parameter}: {x[i]: 1.2f}"

            # Log all objective function values
            for i, of_val in enumerate(objective_values):
                Entry = Entry + f", ObjectiveFunction_{i+1}: {of_val: 1.2f}"

            Entry = Entry + "\n"
            f.write(Entry)
        logger.info(Entry.rstrip())

    def _update_pareto_front(self):
        """Recompute the non-dominated set over all evaluations so far.

        Uses the shared fast non-dominated sorting in
        :func:`TopasMOO.metrics.calculate_dominance_rank` (rank 0 == the Pareto
        front). This drives **intermediate** monitoring plots during a run and
        is written to ``ParetoFront_Running.txt``.

        The **official** final front written to ``ParetoFront.txt`` and used for
        end-of-run figures is the optimizer result (``res.F`` / ``res.X``); see
        ``RunOptimization``. Keeping the two in separate files means neither
        definition can overwrite the other.

        On a resumed run the in-memory history starts empty, so the running
        front covers post-resume evaluations only (see ``_load_eval_cache``).
        The official final front is unaffected.
        """
        if len(self.AllObjectiveFunctionValues) == 0:
            return

        all_objectives = np.array(self.AllObjectiveFunctionValues)
        pareto_indices = np.where(calculate_dominance_rank(all_objectives) == 0)[0]
        self.ParetoObjectives = all_objectives[pareto_indices]

        # Store corresponding decision variables if available
        if len(self.AllDecisionVariables) == len(all_objectives):
            all_decisions = np.array(self.AllDecisionVariables)
            self.ParetoDecisionVars = all_decisions[pareto_indices]

        LogParetoFrontToFile(
            self._ParetoRunningLogFileLoc,
            self.ParetoObjectives,
            self.ParameterNames,
            self.n_objectives,
            ParetoDecisionVars=self.ParetoDecisionVars,
        )

    def _write_final_log_entry(self):
        """Append a short completion summary and Pareto count to the log file."""
        with open(self._LogFileLoc, "a") as f:
            Entry = f'\n{"="*80}\n'
            Entry += f"Optimization Complete: Found {len(self.ParetoObjectives)} Pareto optimal solutions\n"
            if self._n_failed_evaluations:
                Entry += (
                    f"Penalized evaluations (failed or non-finite): "
                    f"{self._n_failed_evaluations}\n"
                )
            Entry += f'{"="*80}\n'
            f.write(Entry)
        logger.info(Entry.strip())

    def _plot_convergence(self):
        """Write objective/parameter convergence and Pareto plots under ``logs``."""
        apply_style(self.intermediate_plot_style, variant=self.publication_variant)
        save_loc = Path(self.BaseDirectory) / self.SimulationName / "logs"

        plot_objective_convergence(
            self._LogFileLoc,
            save_loc / "ConvergencePlot",
            n_objectives=self.n_objectives,
            dpi=INTERMEDIATE_PLOT_DPI,
        )

        plot_parameter_convergence(
            self._LogFileLoc,
            save_loc / "ParameterConvergence",
            parameter_names=self.ParameterNames,
            dpi=INTERMEDIATE_PLOT_DPI,
        )

        if len(self.ParetoObjectives) > 0:
            plot_pareto_front(
                self.ParetoObjectives,
                save_loc / "ParetoFront",
                show_knee_point=True,
                dpi=INTERMEDIATE_PLOT_DPI,
            )

    def GenerateFinalVisualizations(self):
        """Produce publication-style figures under ``logs/FinalResults``."""
        apply_style(self.plot_style, variant=self.publication_variant)

        SaveLoc = (
            Path(self.BaseDirectory) / self.SimulationName / "logs" / "FinalResults"
        )
        GenerateComprehensiveVisualizations(self, SaveLoc, final_plots=self.final_plots)

    def _copy_self(self):
        """Serialize picklable attributes to ``OptimizationSettings.json``.

        Only runs when ``dump_optimization_settings=True``. ``jsonpickle`` is an
        optional dependency (``pip install TopasMOO[settings-dump]``) since
        nothing else in the library needs it.

        :raises ImportError: If ``jsonpickle`` is not installed.
        """
        try:
            import jsonpickle
        except ImportError as exc:
            raise ImportError(
                "dump_optimization_settings=True requires the optional "
                "'jsonpickle' package. Install it with "
                "`pip install TopasMOO[settings-dump]`, or leave "
                "dump_optimization_settings at its default of False."
            ) from exc

        Filename = (
            Path(self.BaseDirectory)
            / Path(self.SimulationName)
            / "OptimizationSettings.json"
        )
        Attributes = jsonpickle.encode(self, unpicklable=True, max_depth=4)
        with open(str(Filename), "w") as f:
            f.write(Attributes)

    def _empty_results_folder(self):
        """Delete contents of the ``Results`` folder (used when ``KeepAllResults`` is False)."""
        ResultsLocation = str(
            Path(self.BaseDirectory) / self.SimulationName / "Results"
        )
        for filename in os.listdir(ResultsLocation):
            file_path = os.path.join(ResultsLocation, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.warning(
                    f"Failed to delete {file_path} from results folder. Reason: {e}. continuing..."
                )

    @abstractmethod
    def RunOptimization(self):
        """Run the optimization; must be implemented by subclasses."""
        pass

    def EvaluateObjectives(self, x_new):
        """Run TOPAS and return the objective vector for ``x_new``.

        The result is cached by decision vector (``logs/EvalCache.jsonl``): a
        design that has already been evaluated (a duplicate within a run, or a
        completed design from a previous run being resumed) returns its stored
        result without re-running TOPAS.

        :param x_new: Dict keyed by ``ParameterNames``, or array of shape ``(n_var,)`` or
            ``(1, n_var)``.

        :returns: One-dimensional ``numpy`` array of length ``n_objectives``. When
            ``n_constraints > 0`` the constraint values for this evaluation are
            stored on ``self._last_constraint_values``.

        :raises InvalidParameterError: If a dict input is missing parameters, or the
            array input is not 1-D/2-D.
        :raises ObjectiveFunctionError: If the objective violates its contract (not a
            list/array, wrong dimensionality, or wrong length). Runtime
            failures (non-zero TOPAS exit, an objective that raises, or
            non-finite values) are penalized unless
            ``on_evaluation_failure="raise"``.

        .. code-block:: python

            Array or dict input (names must match ``ParameterNames``)::

                f = optimizer.EvaluateObjectives(np.array([1.0, 2.0]))
                f = optimizer.EvaluateObjectives({"gap": 1.0, "thickness": 2.0})
        """
        n_var = len(self.ParameterNames)
        if isinstance(x_new, dict):
            missing_params = [name for name in self.ParameterNames if name not in x_new]
            if missing_params:
                raise InvalidParameterError(
                    f"Input dictionary is missing parameters: {missing_params}"
                )
            x = [x_new[param_name] for param_name in self.ParameterNames]
            self.x = np.array(x, dtype=float, ndmin=2)
        else:
            x_arr = np.asarray(x_new, dtype=float)
            if x_arr.ndim == 1:
                self.x = np.expand_dims(x_arr, 0)
            elif x_arr.ndim == 2:
                self.x = x_arr
            else:
                raise InvalidParameterError(
                    "EvaluateObjectives expects a 1D or 2D array-like input."
                )

        # EvaluateObjectives evaluates a single design. Only a (1, n_var) row is
        # supported: a population matrix would be silently flattened into the
        # cache key and then only its first row used downstream. pymoo batches are
        # split into single rows by TopasProblem._evaluate before reaching here.
        if self.x.shape != (1, n_var):
            raise InvalidParameterError(
                f"EvaluateObjectives expects a single design of shape "
                f"({n_var},) or (1, {n_var}). Got input of shape {self.x.shape}. "
                "Pass one design at a time, not a population matrix."
            )

        cache_key = tuple(self.x.flatten().tolist())
        if cache_key in self._eval_cache:
            # Duplicate / resumed design: reuse the stored result, skip TOPAS.
            raw = self._eval_cache[cache_key]
        else:
            self._create_variable_dictionary(self.x)
            self._generate_topas_model()
            if not self.KeepAllResults:
                self._empty_results_folder()
            raw = self._collect_raw_objectives()
            self._eval_cache[cache_key] = raw
            self._persist_evaluation(cache_key, raw)

        objectives = np.asarray(raw[: self.n_objectives], dtype=float)
        if self.n_constraints > 0:
            self._last_constraint_values = np.asarray(
                raw[self.n_objectives :], dtype=float
            )

        self.AllObjectiveFunctionValues.append(objectives)
        self.AllDecisionVariables.append(self.x.flatten().copy())
        self._update_optimization_logs(self.x, objectives)

        # Recompute the Pareto front only when we are about to plot it, rather
        # than on every evaluation: non-dominated sorting is O(n^2) in the
        # number of evaluations accumulated so far.
        if self.evaluation_index % self.plot_frequency == 0:
            self._update_pareto_front()
            self._plot_convergence()

        self.evaluation_index = self.evaluation_index + 1
        self._persist_run_state()

        return objectives

    def _collect_raw_objectives(self):
        """Run TOPAS + the user objective and return a validated ``raw`` vector.

        The returned vector has length ``n_objectives + n_constraints`` (objectives
        first, then constraint values). Runtime failures (non-zero TOPAS exit, an
        objective that raises, or non-finite values) are penalized or re-raised
        per ``on_evaluation_failure``; contract violations always raise.
        """
        expected = self.n_objectives + self.n_constraints
        try:
            self._run_topas_model()
            objective_values = self.TopasObjectiveFunction(
                Path(self.BaseDirectory) / self.SimulationName / "Results",
                self.evaluation_index,
            )
        except TopasExecutionError as exc:
            return self._handle_runtime_failure("TOPAS execution failed", exc)
        except Exception as exc:  # objective raised (e.g. could not read output)
            return self._handle_runtime_failure("objective function raised", exc)

        # --- Contract validation: these are deterministic bugs and always raise.
        if not isinstance(objective_values, (list, np.ndarray)):
            msg = (
                f"TopasObjectiveFunction must return a list or numpy array. "
                f"Got {type(objective_values)} instead."
            )
            logger.error(msg)
            raise ObjectiveFunctionError(msg)
        objective_values = np.asarray(objective_values, dtype=float)
        if objective_values.ndim != 1:
            raise ObjectiveFunctionError(
                f"TopasObjectiveFunction must return a 1D list/array. "
                f"Got array with shape {objective_values.shape}."
            )
        if len(objective_values) != expected:
            what = (
                f"{self.n_objectives} objectives"
                if self.n_constraints == 0
                else f"{self.n_objectives} objectives + {self.n_constraints} constraints"
            )
            msg = (
                f"TopasObjectiveFunction returned {len(objective_values)} values, but {expected} "
                f"were expected ({what})."
            )
            logger.error(msg)
            raise ObjectiveFunctionError(msg)

        # --- Non-finite values are a runtime failure, not a contract bug.
        if not np.isfinite(objective_values).all():
            return self._handle_runtime_failure(
                "objective returned non-finite values", None
            )
        return objective_values

    def _handle_runtime_failure(self, reason, exc):
        """Penalize (default) or re-raise a runtime evaluation failure.

        Returns a worst-case ``raw`` vector when ``on_evaluation_failure`` is
        ``"penalize"``; otherwise re-raises.
        """
        detail = f" ({exc!r})" if exc is not None else ""
        if self.on_evaluation_failure == "raise":
            logger.error("Iteration %d failed: %s%s", self.evaluation_index, reason, detail)
            if exc is not None:
                raise exc
            raise ObjectiveFunctionError(f"{reason} at iteration {self.evaluation_index}")
        self._n_failed_evaluations += 1
        logger.warning("Penalizing iteration %d: %s%s", self.evaluation_index, reason, detail)
        return self._penalty_vector()

    def _penalty_vector(self):
        """Worst-case ``raw`` vector: dominated objectives + infeasible constraints."""
        objectives = np.full(self.n_objectives, self.failure_penalty, dtype=float)
        if self.n_constraints == 0:
            return objectives
        # Positive constraint value => infeasible (pymoo treats g(x) <= 0 feasible).
        constraints = np.full(self.n_constraints, abs(self.failure_penalty), dtype=float)
        return np.concatenate([objectives, constraints])

    def _persist_evaluation(self, cache_key, raw):
        """Append one evaluation to the JSONL evaluation cache (full precision)."""
        try:
            with open(self._eval_cache_loc, "a") as f:
                f.write(
                    json.dumps(
                        {"x": list(cache_key), "raw": [float(v) for v in raw]}
                    )
                    + "\n"
                )
        except Exception as e:
            logger.warning("Could not persist evaluation to cache: %s", e)

    def _load_eval_cache(self):
        """Populate the in-memory evaluation cache from a previous run's JSONL file.

        Records whose ``x``/``raw`` lengths do not match the current optimizer
        configuration (parameter count and ``n_objectives + n_constraints``) are
        skipped: they come from a run with an incompatible problem definition and
        would corrupt the objective/constraint slicing in ``EvaluateObjectives``.

        Accepted records are also replayed into ``AllObjectiveFunctionValues`` /
        ``AllDecisionVariables`` (in file order, which is the original evaluation
        order) so the mid-run front in ``ParetoFront_Running.txt`` spans the whole
        run rather than only post-resume evaluations. A design that is re-asked
        after the resume appends a duplicate point; non-dominated sorting handles
        ties, so this affects only the density of the monitoring scatter.
        """
        if not os.path.isfile(self._eval_cache_loc):
            return
        n_var = len(self.ParameterNames)
        raw_len = self.n_objectives + self.n_constraints
        skipped_incompatible = 0
        with open(self._eval_cache_loc) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = tuple(float(v) for v in rec["x"])
                    raw = np.asarray(rec["raw"], dtype=float)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                    logger.warning("Skipping malformed eval-cache line: %s", e)
                    continue
                if len(key) != n_var or raw.shape != (raw_len,):
                    skipped_incompatible += 1
                    continue
                self._eval_cache[key] = raw
                self.AllObjectiveFunctionValues.append(
                    np.asarray(raw[: self.n_objectives], dtype=float)
                )
                self.AllDecisionVariables.append(np.asarray(key, dtype=float))
        if skipped_incompatible:
            logger.warning(
                "Skipped %d cached evaluation(s) incompatible with the current "
                "configuration (expected %d parameters and %d objective/constraint "
                "values).",
                skipped_incompatible,
                n_var,
                raw_len,
            )
        logger.info("Loaded %d cached evaluations for resume", len(self._eval_cache))

    def _persist_run_state(self) -> None:
        """Write the next ``evaluation_index`` for a clean resume."""
        try:
            with open(self._run_state_loc, "w", encoding="utf-8") as f:
                json.dump({"evaluation_index": int(self.evaluation_index)}, f)
        except OSError as e:
            logger.warning("Could not write run state: %s", e)

    def _restore_evaluation_index(self) -> None:
        """Continue ``evaluation_index`` after ``resume=True``.

        Prefers ``logs/RunState.json`` (stores the next index to use). Falls
        back to one past the maximum ``Iteration`` in ``OptimizationLogs.txt``.
        Without this, resumed runs restart numbering at 0 and can collide with
        existing script names and log rows.
        """
        if os.path.isfile(self._run_state_loc):
            try:
                with open(self._run_state_loc, encoding="utf-8") as f:
                    state = json.load(f)
                restored = int(state["evaluation_index"])
                if restored >= 0:
                    self.evaluation_index = restored
                    logger.info(
                        "Restored evaluation_index=%d from run state",
                        self.evaluation_index,
                    )
                    return
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logger.warning("Could not read run state (%s); trying log file", e)

        if not os.path.isfile(self._LogFileLoc):
            return
        try:
            logged = ReadInMultiObjectiveLogFile(self._LogFileLoc)
            iterations = logged.get("Iteration") or []
            if iterations:
                self.evaluation_index = int(max(iterations)) + 1
                logger.info(
                    "Restored evaluation_index=%d from optimization log",
                    self.evaluation_index,
                )
        except (OSError, MalformedOutputError, ValueError) as e:
            logger.warning("Could not restore evaluation_index from logs: %s", e)

    def _discard_stale_final_front(self) -> None:
        """Remove a previous run's ``ParetoFront.txt`` when resuming.

        Only the end-of-run path writes that file, so on resume it holds the
        *previous* run's final front. Leaving it in place means a resumed run
        that crashes leaves behind a file that reads as this run's official
        result. The running front (``ParetoFront_Running.txt``) is kept.
        """
        try:
            os.remove(self._ParetoLogFileLoc)
        except FileNotFoundError:
            return
        except OSError as e:
            logger.warning("Could not remove stale Pareto front file: %s", e)
            return
        logger.info(
            "Removed stale ParetoFront.txt from the previous run; it will be "
            "rewritten when this run completes."
        )

    def SetUpDirectoryStructure(self):
        """Create simulation subfolders, logs, scripts, results, readme, and test emulator.

        When ``resume=True`` an existing simulation folder is preserved (not
        emptied) and the evaluation cache is loaded, so completed simulations
        are not repeated; missing subfolders are (re)created. Any
        ``ParetoFront.txt`` left by the previous run is removed, since it
        describes that run's final result and would otherwise look authoritative
        until the resumed run finishes and rewrites it.
        """
        FullSimName = Path(self.BaseDirectory) / self.SimulationName
        if not os.path.isdir(FullSimName):
            os.mkdir(FullSimName)

        if self.resume:
            for sub in ("logs", "logs/TopasLogs", "TopasScripts", "Results"):
                os.makedirs(FullSimName / sub, exist_ok=True)
            self._discard_stale_final_front()
            self._load_eval_cache()
            self._restore_evaluation_index()
        else:
            self._empty_simulation_folder()
            if self.dump_optimization_settings:
                self._copy_self()
            os.mkdir(Path(FullSimName) / "logs")
            os.mkdir(Path(FullSimName) / "logs" / "TopasLogs")
            os.mkdir(Path(FullSimName) / "TopasScripts")
            os.mkdir(Path(FullSimName) / "Results")

        if self.ReadMeText:
            with open(FullSimName / "readme.txt", "w") as f:
                f.write(self.ReadMeText)

        if self._testing_mode:
            self._setup_topas_emulator()


class TopasProblem(Problem):
    """pymoo ``Problem`` that evaluates designs via ``TopasMOOBaseClass.EvaluateObjectives``.

    Maps batch ``X`` rows to sequential TOPAS runs on the wrapped optimizer instance.
    """

    def __init__(self, optimizer_instance):
        """Define variable count, bounds, and objective count from a TopasMOO optimizer.

        :param optimizer_instance: Concrete ``TopasMOOBaseClass`` subclass (e.g. NSGA-II).
        """
        self.optimizer = optimizer_instance

        super().__init__(
            n_var=len(optimizer_instance.ParameterNames),
            n_obj=optimizer_instance.n_objectives,
            n_ieq_constr=optimizer_instance.n_constraints,
            xl=optimizer_instance.LowerBounds,
            xu=optimizer_instance.UpperBounds,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """Compute objectives (and constraints) for each row of ``X``.

        Sets ``out['F']`` and, when the optimizer has constraints, ``out['G']``
        (pymoo convention: ``g(x) <= 0`` is feasible).

        :param X: Population matrix of shape ``(pop_size, n_var)``.
        :param out: pymoo output dict; this method sets ``out['F']`` (+ ``out['G']``).
        """
        n_constraints = self.optimizer.n_constraints
        F = []
        G = []
        for x in X:
            objectives = self.optimizer.EvaluateObjectives(x)
            F.append(objectives)
            if n_constraints > 0:
                G.append(
                    np.asarray(self.optimizer._last_constraint_values, dtype=float)
                )

        out["F"] = np.array(F)
        if n_constraints > 0:
            out["G"] = np.array(G)


class _StartPointSampling(FloatRandomSampling):
    """Random initial population with the user's ``start_point`` as one member.

    Replaces the first random individual with the supplied start point, so a
    known-good design participates in the search from generation 0 instead of
    being validated but never used. Runs inside pymoo's seeded sampling flow,
    so seeded runs stay deterministic.
    """

    def __init__(self, start_point):
        super().__init__()
        self.start_point = np.asarray(start_point, dtype=float)

    def _do(self, problem, n_samples, **kwargs):
        X = super()._do(problem, n_samples, **kwargs)
        X[0] = self.start_point
        return X


class NSGAII_Optimizer(TopasMOOBaseClass):
    """NSGA-II (pymoo) for multi-objective TOPAS optimization.

    Uses non-dominated sorting, crowding distance, and tournament selection.
    Additional keyword arguments are forwarded to ``TopasMOOBaseClass``.

    Note on ``pop_size``: the final Pareto front is the non-dominated set of the
    final population, so the number of returned solutions and their objective-space
    coverage are bounded by ``pop_size``. The default of 20 suits quick exploration;
    resolving a real front well usually needs a larger population (at the cost of
    more TOPAS runs per generation).

    :param pop_size: Individuals per generation (default 20).
    :param custom_algorithm: Optional pymoo ``Algorithm``; if omitted, builds ``NSGA2``
        with SBX crossover and PM mutation.
    :param seed: Random seed for reproducibility.
    :param verbose: If True, pymoo prints generation-by-generation progress to stdout.
        Defaults to False (library-friendly).
    :param eliminate_duplicates: If True (default), NSGA-II resamples to avoid
        re-evaluating identical decision vectors -- important when each
        evaluation is an expensive TOPAS simulation.
    :param **kwds: Passed to ``TopasMOOBaseClass.__init__``.
    """

    def __init__(
        self,
        pop_size: int = 20,
        custom_algorithm=None,
        seed: int | None = None,
        verbose: bool = False,
        eliminate_duplicates: bool = True,
        **kwds,
    ):
        """Attach NSGA-II settings and delegate base initialization to the superclass.

        :param pop_size: NSGA-II population size.
        :param custom_algorithm: User-supplied pymoo algorithm, or None for default NSGA2.
        :param seed: RNG seed for the optimization.
        :param verbose: If True, pymoo prints progress on every generation.
        :param eliminate_duplicates: Avoid re-evaluating identical designs (default True).
        :param **kwds: Base-class constructor arguments (directories, params, etc.).
        """
        self.pop_size = pop_size
        self.custom_algorithm = custom_algorithm
        self.seed = seed
        self.verbose = verbose
        self.eliminate_duplicates = eliminate_duplicates

        super().__init__(**kwds)

        if custom_algorithm is None:
            self.algorithm = NSGA2(
                pop_size=pop_size,
                sampling=_StartPointSampling(self.StartingValues),
                crossover=SBX(eta=15, prob=0.9),
                mutation=PM(eta=20),
                eliminate_duplicates=eliminate_duplicates,
            )
        else:
            self.algorithm = custom_algorithm
            logger.info("Using custom algorithm provided by user")

    def RunOptimization(self):
        """Set up folders, run NSGA-II to the generation limit, then finalize outputs.

        Driven by pymoo's ask/tell loop so the algorithm state can be
        checkpointed after every generation (``logs/Checkpoint.pkl``). Together
        with the evaluation cache (``logs/EvalCache.jsonl``) and
        ``logs/RunState.json`` (evaluation counter), this lets a crashed or
        interrupted run be resumed via ``resume=True`` without repeating
        completed TOPAS simulations or resetting iteration numbering.
        (A resumed run's hypervolume history covers only post-resume
        generations; the official final Pareto front is always ``res.F``.)

        :returns: The pymoo ``Result`` object (also stored as ``self.res``).
            ``res.F`` / ``res.X`` are the official final Pareto set and match
            ``ParetoFront.txt`` and end-of-run figures. Mid-run monitoring uses
            the ND set over all evaluations so far (``ParetoFront_Running.txt``).
        """
        self.SetUpDirectoryStructure()

        # Create pymoo problem
        problem = TopasProblem(self)

        # Set up termination criterion
        termination = get_termination("n_gen", self.n_generations)

        algorithm = self._load_state_checkpoint(problem)
        if algorithm is None:
            algorithm = self.algorithm
            algorithm.setup(
                problem,
                termination=termination,
                seed=self.seed,
                verbose=self.verbose,
            )
            logger.info(
                "Starting NSGA-II optimization with %d individuals for %d generations",
                self.pop_size,
                self.n_generations,
            )
        else:
            logger.info(
                "Resuming NSGA-II from checkpoint (generation %s)", algorithm.n_gen
            )

        # Ask/tell loop: one generation per iteration, checkpointed each step.
        gen_populations = []
        while algorithm.has_next():
            algorithm.next()
            gen_populations.append(np.asarray(algorithm.pop.get("F")).copy())
            self._save_state_checkpoint(algorithm)

        self.res = algorithm.result()

        self._extract_optimization_history(gen_populations)

        # Official final front: the non-dominated set of the optimizer's final
        # population (``res.F`` / ``res.X``). Intermediate monitoring during the
        # run uses the ND set over *all evaluations* and writes
        # ``ParetoFront_Running.txt``; only this end-of-run path writes
        # ``ParetoFront.txt``, so the two definitions never share a file.
        self.ParetoObjectives = np.atleast_2d(np.asarray(self.res.F, dtype=float))
        self.ParetoDecisionVars = np.atleast_2d(np.asarray(self.res.X, dtype=float))
        LogParetoFrontToFile(
            self._ParetoLogFileLoc,
            self.ParetoObjectives,
            self.ParameterNames,
            self.n_objectives,
            ParetoDecisionVars=self.ParetoDecisionVars,
        )
        self._write_final_log_entry()
        self._persist_run_state()

        logger.info(
            "Optimization complete. Found %d solutions in the Pareto front.",
            len(self.ParetoObjectives),
        )

        # Generate final visualizations
        self._plot_convergence()
        self.GenerateFinalVisualizations()

        return self.res

    def _extract_optimization_history(self, populations=None):
        """Fill ``HypervolumeHistory`` and ``PopulationHistory`` from per-generation data.

        :param populations: Optional list of per-generation objective matrices
            (each ``(pop_size, n_obj)``), as collected by the ask/tell loop.
            The built-in NSGA-II path always passes this explicitly. If
            ``None``, it falls back to ``self.res.history`` *when populated*
            -- only the case for a custom algorithm run with pymoo's
            ``save_history=True``; otherwise no history is recorded.

        The hypervolume reference point is derived from the objective values
        actually observed across all generations (per-objective nadir plus a 10%
        margin of the observed range), so the indicator is meaningful regardless
        of the objectives' scale or sign rather than assuming values in ``[0, 1]``.
        Logs a warning and records ``0.0`` for any generation whose hypervolume
        cannot be computed.
        """
        if not populations:
            if hasattr(self.res, "history") and self.res.history:
                populations = [algo.pop.get("F") for algo in self.res.history]
            else:
                logger.warning("No history available in optimization results")
                return

        from pymoo.indicators.hv import HV

        # Collect every generation's population objectives once.
        self.PopulationHistory = [
            (gen_idx, np.asarray(pop).copy()) for gen_idx, pop in enumerate(populations)
        ]
        populations = [np.asarray(pop) for pop in populations]

        # Reference point: worse (i.e. larger, for minimization) than every
        # observed objective value, with a margin proportional to the observed
        # range. Using a single fixed reference keeps the per-generation
        # hypervolumes comparable across the run.
        all_objectives = np.vstack(populations)
        ideal = all_objectives.min(axis=0)
        nadir = all_objectives.max(axis=0)
        span = nadir - ideal
        span[span == 0] = 1.0
        ref_point = nadir + 0.1 * span
        hv_indicator = HV(ref_point=ref_point)

        self.HypervolumeHistory = []
        for gen_idx, pop_objectives in enumerate(populations):
            try:
                self.HypervolumeHistory.append(hv_indicator(pop_objectives))
            except Exception as e:
                logger.warning(
                    f"Could not compute hypervolume for generation {gen_idx}: {e}"
                )
                self.HypervolumeHistory.append(0.0)

        logger.info(f"Extracted history for {len(self.HypervolumeHistory)} generations")
        if self.HypervolumeHistory:
            logger.info(f"Final hypervolume: {self.HypervolumeHistory[-1]:.6f}")

    def _save_state_checkpoint(self, algorithm):
        """Pickle the NSGA-II algorithm state for exact resume (best-effort).

        The problem and the (potentially large) history are detached before
        pickling -- the problem holds the user-imported modules and is re-attached
        on resume, and the full (x, F) record lives in the evaluation cache. A
        failure here is logged and ignored: the evaluation cache alone is enough
        to resume without repeating completed simulations.
        """
        problem, history = algorithm.problem, algorithm.history
        algorithm.problem, algorithm.history = None, []
        try:
            with open(self._checkpoint_loc, "wb") as f:
                pickle.dump(algorithm, f)
        except Exception as e:
            logger.warning("Could not write algorithm checkpoint: %s", e)
        finally:
            algorithm.problem, algorithm.history = problem, history

    def _load_state_checkpoint(self, problem):
        """Return a resumed NSGA-II algorithm from ``Checkpoint.pkl``, or ``None``.

        Only attempts to load when ``resume=True`` and a checkpoint exists. On
        any failure it returns ``None`` so the GA restarts from scratch -- the
        warm evaluation cache still prevents completed simulations from re-running.
        """
        if not (self.resume and os.path.isfile(self._checkpoint_loc)):
            return None
        try:
            with open(self._checkpoint_loc, "rb") as f:
                algorithm = pickle.load(f)
            algorithm.problem = problem
            return algorithm
        except Exception as e:
            logger.warning(
                "Could not load checkpoint (%s); restarting the GA "
                "(cached evaluations are still reused).",
                e,
            )
            return None
