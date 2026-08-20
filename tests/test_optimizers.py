"""
Comprehensive tests for TopasMOO optimizers

Tests cover:
- Multi-objective requirements enforcement
- NSGA-II optimizer initialization and configuration
- Parameter validation and bounds checking
- Directory structure creation
- Pareto front identification and tracking
- Topas problem class
- Logging functionality
"""
import builtins
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path to import TopasMOO
sys.path.insert(0, str(Path(__file__).parent.parent))

from TopasMOO import optimizers as tmo
from TopasMOO.exceptions import InvalidParameterError

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def basic_params():
    """Basic valid optimization parameters for testing"""
    return {
        "ParameterNames": ["param1", "param2"],
        "UpperBounds": np.array([10.0, 10.0]),
        "LowerBounds": np.array([0.0, 0.0]),
        "start_point": np.array([5.0, 5.0]),
        "n_iterations": 2,
        "n_objectives": 2,
    }


@pytest.fixture
def single_param():
    """Single parameter optimization params"""
    return {
        "ParameterNames": ["param1"],
        "UpperBounds": np.array([10.0]),
        "LowerBounds": np.array([0.0]),
        "start_point": np.array([5.0]),
        "n_iterations": 2,
        "n_objectives": 2,
    }


@pytest.fixture
def three_objective_params():
    """Three-objective optimization params"""
    return {
        "ParameterNames": ["x1", "x2", "x3"],
        "UpperBounds": np.array([1.0, 1.0, 1.0]),
        "LowerBounds": np.array([0.0, 0.0, 0.0]),
        "start_point": np.array([0.5, 0.5, 0.5]),
        "n_iterations": 5,
        "n_objectives": 3,
    }


# temp_dir / opt_dir come from tests/conftest.py.


# ============================================================================
# Multi-Objective Requirements Tests
# ============================================================================


class TestMultiObjectiveRequirements:
    """Test that TopasMOO correctly enforces multi-objective requirements"""

    def test_requires_n_objectives(self, temp_dir, opt_dir):
        """Test that n_objectives must be specified"""
        optimization_params = {
            "ParameterNames": ["param1"],
            "UpperBounds": np.array([10]),
            "LowerBounds": np.array([0]),
            "start_point": np.array([5]),
            "n_iterations": 2
            # Missing n_objectives - should raise error
        }

        with pytest.raises(InvalidParameterError):
            tmo.NSGAII_Optimizer(
                optimization_params=optimization_params,
                BaseDirectory=temp_dir,
                SimulationName="test",
                OptimizationDirectory=opt_dir,
                TopasLocation="testing_mode",
            )

    def test_requires_multiple_objectives(self, temp_dir, opt_dir):
        """Test that n_objectives must be >= 2"""
        optimization_params = {
            "ParameterNames": ["param1"],
            "UpperBounds": np.array([10]),
            "LowerBounds": np.array([0]),
            "start_point": np.array([5]),
            "n_iterations": 2,
            "n_objectives": 1,  # Single objective - should raise error
        }

        with pytest.raises(InvalidParameterError):
            tmo.NSGAII_Optimizer(
                optimization_params=optimization_params,
                BaseDirectory=temp_dir,
                SimulationName="test",
                OptimizationDirectory=opt_dir,
                TopasLocation="testing_mode",
            )

    def test_n_objectives_zero_fails(self, temp_dir, opt_dir):
        """Test that n_objectives=0 fails"""
        optimization_params = {
            "ParameterNames": ["param1"],
            "UpperBounds": np.array([10]),
            "LowerBounds": np.array([0]),
            "start_point": np.array([5]),
            "n_iterations": 2,
            "n_objectives": 0,
        }

        with pytest.raises(InvalidParameterError):
            tmo.NSGAII_Optimizer(
                optimization_params=optimization_params,
                BaseDirectory=temp_dir,
                SimulationName="test",
                OptimizationDirectory=opt_dir,
                TopasLocation="testing_mode",
            )

    def test_n_objectives_negative_fails(self, temp_dir, opt_dir):
        """Test that negative n_objectives fails"""
        optimization_params = {
            "ParameterNames": ["param1"],
            "UpperBounds": np.array([10]),
            "LowerBounds": np.array([0]),
            "start_point": np.array([5]),
            "n_iterations": 2,
            "n_objectives": -2,
        }

        with pytest.raises(InvalidParameterError):
            tmo.NSGAII_Optimizer(
                optimization_params=optimization_params,
                BaseDirectory=temp_dir,
                SimulationName="test",
                OptimizationDirectory=opt_dir,
                TopasLocation="testing_mode",
            )

    def test_accepts_two_objectives(self, temp_dir, basic_params, opt_dir):
        """Test that exactly 2 objectives is accepted"""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_2obj",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )
        assert optimizer.n_objectives == 2

    def test_accepts_three_objectives(self, temp_dir, three_objective_params, opt_dir):
        """Test that 3 objectives is accepted"""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=three_objective_params,
            BaseDirectory=temp_dir,
            SimulationName="test_3obj",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )
        assert optimizer.n_objectives == 3

    def test_accepts_many_objectives(self, temp_dir, opt_dir):
        """Test that many objectives (5+) is accepted"""
        optimization_params = {
            "ParameterNames": ["p1", "p2", "p3"],
            "UpperBounds": np.array([1.0, 1.0, 1.0]),
            "LowerBounds": np.array([0.0, 0.0, 0.0]),
            "start_point": np.array([0.5, 0.5, 0.5]),
            "n_iterations": 2,
            "n_objectives": 5,
        }

        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=optimization_params,
            BaseDirectory=temp_dir,
            SimulationName="test_5obj",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )
        assert optimizer.n_objectives == 5


# ============================================================================
# NSGA-II Optimizer Tests
# ============================================================================


class TestNSGAIIOptimizer:
    """Test NSGA-II optimizer functionality"""

    def test_initialization(self, temp_dir, basic_params, opt_dir):
        """Test that NSGA-II optimizer initializes correctly"""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_init",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
            pop_size=4,
        )

        assert optimizer.n_objectives == 2
        assert len(optimizer.ParameterNames) == 2
        assert optimizer.pop_size == 4

    def test_default_pop_size(self, temp_dir, basic_params, opt_dir):
        """Test default population size"""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_default_pop",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )

        assert optimizer.pop_size == 20  # Default value

    def test_custom_pop_size(self, temp_dir, basic_params, opt_dir):
        """Test custom population size"""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_custom_pop",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
            pop_size=50,
        )

        assert optimizer.pop_size == 50

    def test_seed_reproducibility(self, temp_dir, basic_params, opt_dir):
        """Test that seed can be set for reproducibility"""
        optimizer1 = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_seed1",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
            seed=42,
        )

        optimizer2 = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_seed2",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
            seed=42,
        )

        assert optimizer1.seed == optimizer2.seed == 42

    def test_stores_parameter_names(self, temp_dir, basic_params, opt_dir):
        """Test that parameter names are stored"""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_params",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )

        assert optimizer.ParameterNames == ["param1", "param2"]

    def test_stores_bounds(self, temp_dir, basic_params, opt_dir):
        """Test that bounds are stored correctly"""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_bounds_store",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )

        np.testing.assert_array_equal(optimizer.UpperBounds, np.array([10.0, 10.0]))
        np.testing.assert_array_equal(optimizer.LowerBounds, np.array([0.0, 0.0]))


# ============================================================================
# NSGA-III Optimizer Tests
# ============================================================================


class TestNSGAIIIOptimizer:
    """Test NSGA-III optimizer initialization and reference directions."""

    def test_initialization_with_two_objectives(self, temp_dir, basic_params, opt_dir):
        optimizer = tmo.NSGAIII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_nsga3_2obj",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
            pop_size=4,
            ref_dir_partitions=3,
        )

        assert optimizer.n_objectives == 2
        assert optimizer.ref_dir_dim == 2
        assert optimizer.algorithm.ref_dirs.shape == (4, 2)
        np.testing.assert_allclose(optimizer.algorithm.ref_dirs.sum(axis=1), 1.0)

    def test_initialization_with_three_objectives(
        self, temp_dir, three_objective_params, opt_dir
    ):
        optimizer = tmo.NSGAIII_Optimizer(
            optimization_params=three_objective_params,
            BaseDirectory=temp_dir,
            SimulationName="test_nsga3_3obj",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
            pop_size=6,
            ref_dir_partitions=2,
        )

        assert optimizer.n_objectives == 3
        assert optimizer.ref_dir_dim == 3
        assert optimizer.algorithm.ref_dirs.shape == (6, 3)
        np.testing.assert_allclose(optimizer.algorithm.ref_dirs.sum(axis=1), 1.0)

    def test_default_pop_size_is_derived_from_reference_directions(
        self, temp_dir, basic_params, opt_dir
    ):
        optimizer = tmo.NSGAIII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_nsga3_derived_pop_size",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )

        assert len(optimizer.ref_dirs) == 13
        assert optimizer.pop_size == 13
        assert optimizer.algorithm.pop_size == 13

    def test_explicit_pop_size_may_exceed_reference_direction_count(
        self, temp_dir, basic_params, opt_dir
    ):
        optimizer = tmo.NSGAIII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_nsga3_larger_pop_size",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
            pop_size=6,
            ref_dir_partitions=3,
        )

        assert len(optimizer.ref_dirs) == 4
        assert optimizer.pop_size == 6
        assert optimizer.algorithm.pop_size == 6

    def test_explicit_pop_size_smaller_than_reference_direction_count_fails(
        self, temp_dir, three_objective_params, opt_dir
    ):
        with pytest.raises(
            InvalidParameterError,
            match=r"pop_size \(5\).*reference directions \(6\)",
        ):
            tmo.NSGAIII_Optimizer(
                optimization_params=three_objective_params,
                BaseDirectory=temp_dir,
                SimulationName="test_nsga3_small_pop_size",
                OptimizationDirectory=opt_dir,
                TopasLocation="testing_mode",
                Overwrite=True,
                pop_size=5,
                ref_dir_partitions=2,
            )


# ============================================================================
# Optimization History Tests
# ============================================================================


class TestOptimizationHistory:
    """Test shared history normalization used by NSGA-II and NSGA-III."""

    @staticmethod
    def _make_optimizer(optimizer_class, temp_dir, basic_params, opt_dir):
        algorithm_options = {"pop_size": 4}
        if optimizer_class is tmo.NSGAIII_Optimizer:
            algorithm_options["ref_dir_partitions"] = 3
        return optimizer_class(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName=f"history_{optimizer_class.__name__}",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
            **algorithm_options,
        )

    @pytest.mark.parametrize(
        "optimizer_class", [tmo.NSGAII_Optimizer, tmo.NSGAIII_Optimizer]
    )
    @pytest.mark.parametrize("stack_generations", [False, True])
    def test_numpy_history_is_normalized(
        self,
        optimizer_class,
        stack_generations,
        temp_dir,
        basic_params,
        opt_dir,
    ):
        optimizer = self._make_optimizer(
            optimizer_class, temp_dir, basic_params, opt_dir
        )
        first = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
        populations = np.stack((first, first * 0.9)) if stack_generations else first

        optimizer._extract_optimization_history(populations)

        expected_generations = 2 if stack_generations else 1
        assert len(optimizer.PopulationHistory) == expected_generations
        assert len(optimizer.HypervolumeHistory) == expected_generations
        np.testing.assert_array_equal(optimizer.PopulationHistory[0][1], first)

    @pytest.mark.parametrize(
        ("populations", "message"),
        [
            (np.ones(4), "array shape"),
            (np.ones((4, 3)), "2 objective columns"),
            (np.ones((3, 2)), "4 rows"),
            (
                np.array(
                    [[np.nan, 1.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]]
                ),
                "non-finite",
            ),
        ],
    )
    def test_malformed_history_raises_clear_error(
        self, populations, message, temp_dir, basic_params, opt_dir
    ):
        optimizer = self._make_optimizer(
            tmo.NSGAII_Optimizer, temp_dir, basic_params, opt_dir
        )

        with pytest.raises(ValueError, match=message):
            optimizer._extract_optimization_history(populations)


# ============================================================================
# Parameter Bounds Validation Tests
# ============================================================================


class TestParameterBoundsValidation:
    """Test that parameter bounds are validated correctly"""

    def test_start_point_above_upper_bound_fails(self, temp_dir, opt_dir):
        """Test that starting point above upper bound fails"""
        optimization_params = {
            "ParameterNames": ["param1"],
            "UpperBounds": np.array([10]),
            "LowerBounds": np.array([0]),
            "start_point": np.array([15]),  # Outside bounds!
            "n_iterations": 2,
            "n_objectives": 2,
        }

        with pytest.raises(InvalidParameterError):
            tmo.NSGAII_Optimizer(
                optimization_params=optimization_params,
                BaseDirectory=temp_dir,
                SimulationName="test_bounds",
                OptimizationDirectory=opt_dir,
                TopasLocation="testing_mode",
                Overwrite=True,
            )

    def test_start_point_below_lower_bound_fails(self, temp_dir, opt_dir):
        """Test that starting point below lower bound fails"""
        optimization_params = {
            "ParameterNames": ["param1"],
            "UpperBounds": np.array([10]),
            "LowerBounds": np.array([0]),
            "start_point": np.array([-5]),  # Below lower bound
            "n_iterations": 2,
            "n_objectives": 2,
        }

        with pytest.raises(InvalidParameterError):
            tmo.NSGAII_Optimizer(
                optimization_params=optimization_params,
                BaseDirectory=temp_dir,
                SimulationName="test_lower_bound",
                OptimizationDirectory=opt_dir,
                TopasLocation="testing_mode",
                Overwrite=True,
            )

    def test_start_point_at_bounds_ok(self, temp_dir, opt_dir):
        """Test that starting point exactly at bounds is allowed"""
        optimization_params = {
            "ParameterNames": ["param1", "param2"],
            "UpperBounds": np.array([10, 10]),
            "LowerBounds": np.array([0, 0]),
            "start_point": np.array([0, 10]),  # At exact bounds
            "n_iterations": 2,
            "n_objectives": 2,
        }

        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=optimization_params,
            BaseDirectory=temp_dir,
            SimulationName="test_at_bounds",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )

        # The bounds should be stored correctly
        np.testing.assert_array_equal(optimizer.LowerBounds, np.array([0, 0]))
        np.testing.assert_array_equal(optimizer.UpperBounds, np.array([10, 10]))

    def test_multiple_params_one_out_of_bounds(self, temp_dir, opt_dir):
        """Test that if any parameter is out of bounds, it fails"""
        optimization_params = {
            "ParameterNames": ["param1", "param2", "param3"],
            "UpperBounds": np.array([10, 10, 10]),
            "LowerBounds": np.array([0, 0, 0]),
            "start_point": np.array([5, 15, 5]),  # param2 out of bounds
            "n_iterations": 2,
            "n_objectives": 2,
        }

        with pytest.raises(InvalidParameterError):
            tmo.NSGAII_Optimizer(
                optimization_params=optimization_params,
                BaseDirectory=temp_dir,
                SimulationName="test_one_bad",
                OptimizationDirectory=opt_dir,
                TopasLocation="testing_mode",
                Overwrite=True,
            )

    def test_lower_greater_than_upper_fails(self, temp_dir, opt_dir):
        """Test that lower bound > upper bound fails"""
        optimization_params = {
            "ParameterNames": ["param1"],
            "UpperBounds": np.array([5]),
            "LowerBounds": np.array([10]),  # Greater than upper!
            "start_point": np.array([7]),
            "n_iterations": 2,
            "n_objectives": 2,
        }

        # This should fail during initialization
        with pytest.raises(InvalidParameterError):
            tmo.NSGAII_Optimizer(
                optimization_params=optimization_params,
                BaseDirectory=temp_dir,
                SimulationName="test_reversed_bounds",
                OptimizationDirectory=opt_dir,
                TopasLocation="testing_mode",
                Overwrite=True,
            )


# ============================================================================
# Directory Structure Tests
# ============================================================================


class TestDirectoryStructure:
    """Test that directory structure is created correctly"""

    def test_creates_directories(self, temp_dir, basic_params, opt_dir):
        """Test that all required directories are created"""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_dirs",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )

        optimizer.SetUpDirectoryStructure()

        sim_dir = Path(temp_dir) / "test_dirs"
        assert (sim_dir / "logs").exists()
        assert (sim_dir / "logs" / "TopasLogs").exists()
        assert (sim_dir / "TopasScripts").exists()
        assert (sim_dir / "Results").exists()

    def test_creates_settings_file_when_requested(self, temp_dir, basic_params, opt_dir):
        """Opt-in jsonpickle dump writes OptimizationSettings.json."""
        pytest.importorskip("jsonpickle")  # optional 'settings-dump' extra
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_settings",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
            dump_optimization_settings=True,
        )

        optimizer.SetUpDirectoryStructure()

        settings_file = Path(temp_dir) / "test_settings" / "OptimizationSettings.json"
        assert settings_file.exists()

        with open(settings_file, "r") as f:
            content = f.read()
            assert len(content) > 0

    def test_default_skips_settings_file(self, temp_dir, basic_params, opt_dir):
        """Settings dump is off by default (not required for resume)."""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_no_settings",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )

        optimizer.SetUpDirectoryStructure()

        settings_file = Path(temp_dir) / "test_no_settings" / "OptimizationSettings.json"
        assert not settings_file.exists()

    def test_settings_dump_without_jsonpickle_raises_actionable_error(
        self, temp_dir, basic_params, opt_dir, monkeypatch
    ):
        """Missing optional dep names the extra to install, not just 'jsonpickle'."""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_missing_jsonpickle",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
            dump_optimization_settings=True,
        )

        real_import = builtins.__import__

        def _no_jsonpickle(name, *args, **kwargs):
            if name == "jsonpickle":
                raise ImportError("No module named 'jsonpickle'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_jsonpickle)

        with pytest.raises(ImportError, match=r"settings-dump"):
            optimizer.SetUpDirectoryStructure()

    def test_overwrite_clears_directories(self, temp_dir, basic_params, opt_dir):
        """Test that Overwrite=True clears existing directories"""
        sim_dir = Path(temp_dir) / "test_overwrite"
        os.makedirs(sim_dir / "logs", exist_ok=True)

        # Create a dummy file
        dummy_file = sim_dir / "logs" / "dummy.txt"
        with open(dummy_file, "w") as f:
            f.write("test")

        assert dummy_file.exists()

        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_overwrite",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )

        optimizer.SetUpDirectoryStructure()

        # Original file should be gone
        assert not dummy_file.exists()


# ============================================================================
# Pareto Front Tests
# ============================================================================


class TestParetoFrontTracking:
    """Test Pareto front identification and tracking"""

    def test_pareto_front_update_basic(self, temp_dir, basic_params, opt_dir):
        """Test that Pareto front is correctly identified"""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_pareto",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )

        # Set up directory structure first
        optimizer.SetUpDirectoryStructure()

        # Add some test solutions
        optimizer.AllObjectiveFunctionValues = [
            np.array([1.0, 5.0]),  # Pareto optimal
            np.array([2.0, 3.0]),  # Pareto optimal
            np.array([3.0, 2.0]),  # Pareto optimal
            np.array([4.0, 4.0]),  # Dominated by (2,3) and (3,2)
        ]

        optimizer._update_pareto_front()

        # Should identify 3 non-dominated solutions
        assert len(optimizer.ParetoObjectives) == 3

    def test_pareto_front_single_solution(self, temp_dir, basic_params, opt_dir):
        """Test Pareto front with a single solution"""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_single",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )

        optimizer.SetUpDirectoryStructure()

        optimizer.AllObjectiveFunctionValues = [np.array([1.0, 1.0])]

        optimizer._update_pareto_front()

        assert len(optimizer.ParetoObjectives) == 1

    def test_pareto_front_all_dominated(self, temp_dir, basic_params, opt_dir):
        """Test when one solution dominates all others"""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_dominated",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )

        optimizer.SetUpDirectoryStructure()

        optimizer.AllObjectiveFunctionValues = [
            np.array([1.0, 1.0]),  # Dominates all others
            np.array([2.0, 2.0]),
            np.array([3.0, 3.0]),
            np.array([4.0, 4.0]),
        ]

        optimizer._update_pareto_front()

        # Only the dominant solution should be on the front
        assert len(optimizer.ParetoObjectives) == 1
        np.testing.assert_array_equal(
            optimizer.ParetoObjectives[0], np.array([1.0, 1.0])
        )

    def test_pareto_front_none_dominated(self, temp_dir, basic_params, opt_dir):
        """Test when no solution dominates another"""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_none_dom",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )

        optimizer.SetUpDirectoryStructure()

        optimizer.AllObjectiveFunctionValues = [
            np.array([1.0, 4.0]),
            np.array([2.0, 3.0]),
            np.array([3.0, 2.0]),
            np.array([4.0, 1.0]),
        ]

        optimizer._update_pareto_front()

        # All solutions are non-dominated
        assert len(optimizer.ParetoObjectives) == 4

    def test_pareto_front_three_objectives(
        self, temp_dir, three_objective_params, opt_dir
    ):
        """Test Pareto front with 3 objectives"""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=three_objective_params,
            BaseDirectory=temp_dir,
            SimulationName="test_3obj_pareto",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )

        optimizer.SetUpDirectoryStructure()

        # In 3D, [2,2,2] is NOT dominated by any of the others:
        # - [1,2,3] is worse in obj3
        # - [2,1,3] is worse in obj3
        # - [3,2,1] is worse in obj1
        # So we need clearly dominated solutions:
        optimizer.AllObjectiveFunctionValues = [
            np.array([1.0, 1.0, 1.0]),  # Pareto optimal - best overall
            np.array([2.0, 1.0, 1.0]),  # Dominated by above
            np.array([1.0, 2.0, 1.0]),  # Dominated by above
            np.array([1.0, 1.0, 2.0]),  # Dominated by above
        ]

        optimizer._update_pareto_front()

        # Only [1,1,1] should be on the front
        assert len(optimizer.ParetoObjectives) == 1


# ============================================================================
# TopasProblem Class Tests
# ============================================================================


class TestTopasProblem:
    """Test the TopasProblem class used by pymoo"""

    def test_problem_creation(self, temp_dir, basic_params, opt_dir):
        """Test that TopasProblem is created correctly"""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_problem",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )

        problem = tmo.TopasProblem(optimizer)

        assert problem.n_var == 2  # Number of parameters
        assert problem.n_obj == 2  # Number of objectives

    def test_problem_bounds(self, temp_dir, basic_params, opt_dir):
        """Test that TopasProblem has correct bounds"""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_problem_bounds",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )

        problem = tmo.TopasProblem(optimizer)

        np.testing.assert_array_equal(problem.xl, np.array([0.0, 0.0]))
        np.testing.assert_array_equal(problem.xu, np.array([10.0, 10.0]))


# ============================================================================
# Iteration Counter Tests
# ============================================================================


class TestIterationTracking:
    """Test iteration counting and tracking"""

    def test_iteration_starts_at_zero(self, temp_dir, basic_params, opt_dir):
        """Test that iteration counter starts at 0"""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_iteration",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )

        assert optimizer.evaluation_index == 0

    def test_max_iterations_stored(self, temp_dir, basic_params, opt_dir):
        """Test that max iterations is stored"""
        basic_params["n_iterations"] = 100

        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_max_iter",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )

        assert optimizer.n_generations == 100

    def test_missing_n_generations_raises_invalid_parameter(self, temp_dir, basic_params, opt_dir):
        """Shared algorithm-step key is required for NSGA-II and MOBO."""
        del basic_params["n_iterations"]
        with pytest.raises(InvalidParameterError, match="n_generations"):
            tmo.NSGAII_Optimizer(
                optimization_params=basic_params,
                BaseDirectory=temp_dir,
                SimulationName="test_missing_gens",
                OptimizationDirectory=opt_dir,
                TopasLocation="testing_mode",
                Overwrite=True,
            )


# ============================================================================
# Logging Tests
# ============================================================================


class TestOptimizerLogging:
    """Test optimizer logging functionality"""

    def test_logs_directory_created(self, temp_dir, basic_params, opt_dir):
        """Test that logs directory is created"""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_log",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )

        optimizer.SetUpDirectoryStructure()

        logs_dir = Path(temp_dir) / "test_log" / "logs"
        assert logs_dir.exists()


# ============================================================================
# Testing Mode Tests
# ============================================================================


class TestTestingMode:
    """Test the testing_mode functionality"""

    def test_testing_mode_creates_emulator(self, temp_dir, basic_params, opt_dir):
        """Test that testing_mode creates a topas emulator"""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_emulator",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )

        optimizer.SetUpDirectoryStructure()

        # Emulator should be created in bin directory
        emulator_path = Path(temp_dir) / "test_emulator" / "bin" / "topas"
        assert emulator_path.exists()


# ============================================================================
# Data Storage Tests
# ============================================================================


class TestDataStorage:
    """Test that optimization data is stored correctly"""

    def test_stores_all_objectives(self, temp_dir, basic_params, opt_dir):
        """Test that all objective values are tracked"""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_obj_storage",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )

        assert hasattr(optimizer, "AllObjectiveFunctionValues")
        assert isinstance(optimizer.AllObjectiveFunctionValues, list)

    def test_stores_n_objectives(self, temp_dir, basic_params, opt_dir):
        """Test that n_objectives is stored"""
        optimizer = tmo.NSGAII_Optimizer(
            optimization_params=basic_params,
            BaseDirectory=temp_dir,
            SimulationName="test_n_obj_storage",
            OptimizationDirectory=opt_dir,
            TopasLocation="testing_mode",
            Overwrite=True,
        )

        assert optimizer.n_objectives == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

