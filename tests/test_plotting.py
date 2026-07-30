"""
Plotting, metrics, and log I/O tests for TopasMOO.

Tests cover:
- All Pareto front visualization functions (2D, 3D, projections)
- Parallel coordinates plots
- Petal diagrams
- Convergence plotting
- Metric calculations (knee point, crowding distance, dominance rank)
- Log file I/O
- Comprehensive visualization generator
- Edge cases and error handling
"""
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from TopasMOO.io import LogParetoFrontToFile, ReadInMultiObjectiveLogFile
from TopasMOO.metrics import (
    calculate_crowding_distance,
    calculate_dominance_rank,
    calculate_knee_point,
)
from TopasMOO.plotting import (
    plot_decision_heatmap,
    plot_objective_convergence,
    plot_parallel_coordinates,
    plot_parameter_convergence,
    plot_parameter_objective_correlation,
    plot_pareto_front_2d,
    plot_pareto_front_3d,
    plot_pareto_front_projections,
    plot_petal_diagram_multi,
    plot_petal_diagram_single,
)
from TopasMOO.plotting.style import apply_style

# ============================================================================
# Fixtures
# ============================================================================


# temp_dir comes from tests/conftest.py.


@pytest.fixture
def pareto_2d():
    """Synthetic 2D Pareto front data"""
    return np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.5], [5.0, 1.0]])


@pytest.fixture
def pareto_3d():
    """Synthetic 3D Pareto front data"""
    return np.array(
        [
            [1.0, 5.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 2.0, 2.0],
            [4.0, 1.5, 1.5],
            [5.0, 1.0, 1.0],
        ]
    )


@pytest.fixture
def pareto_4d():
    """Synthetic 4D Pareto front data (high-dimensional)"""
    np.random.seed(42)
    return np.random.rand(15, 4) * 10


@pytest.fixture
def pareto_5d():
    """Synthetic 5D Pareto front data"""
    np.random.seed(123)
    return np.random.rand(20, 5) * 10


@pytest.fixture
def decision_vars():
    """Synthetic decision variable data"""
    return np.array([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1]])


@pytest.fixture
def sample_log_file(temp_dir):
    """Create a sample log file for testing"""
    log_path = Path(temp_dir) / "test_log.txt"
    with open(log_path, "w") as f:
        f.write(
            "Iteration: 0, param1: 5.00, param2: 5.00, ObjectiveFunction_1: 10.00, ObjectiveFunction_2: 15.00\n"
        )
        f.write(
            "Iteration: 1, param1: 4.50, param2: 5.50, ObjectiveFunction_1: 9.00, ObjectiveFunction_2: 14.00\n"
        )
        f.write(
            "Iteration: 2, param1: 4.00, param2: 6.00, ObjectiveFunction_1: 8.00, ObjectiveFunction_2: 13.00\n"
        )
        f.write(
            "Iteration: 3, param1: 3.50, param2: 6.50, ObjectiveFunction_1: 7.00, ObjectiveFunction_2: 12.00\n"
        )
        f.write(
            "Iteration: 4, param1: 3.00, param2: 7.00, ObjectiveFunction_1: 6.00, ObjectiveFunction_2: 11.00\n"
        )
    return log_path


@pytest.fixture
def sample_log_file_3obj(temp_dir):
    """Create a sample log file with 3 objectives"""
    log_path = Path(temp_dir) / "test_log_3obj.txt"
    with open(log_path, "w") as f:
        f.write(
            "Iteration: 0, x1: 0.50, x2: 0.50, ObjectiveFunction_1: 10.00, ObjectiveFunction_2: 15.00, ObjectiveFunction_3: 12.00\n"
        )
        f.write(
            "Iteration: 1, x1: 0.40, x2: 0.60, ObjectiveFunction_1: 9.00, ObjectiveFunction_2: 14.00, ObjectiveFunction_3: 11.00\n"
        )
        f.write(
            "Iteration: 2, x1: 0.30, x2: 0.70, ObjectiveFunction_1: 8.00, ObjectiveFunction_2: 13.00, ObjectiveFunction_3: 10.00\n"
        )
    return log_path


# ============================================================================
# 2D Pareto Front Tests
# ============================================================================


class TestParetoFront2D:
    """Test 2D Pareto front plotting"""

    def test_basic_2d_plot(self, temp_dir, pareto_2d):
        """Test basic 2D Pareto front plot creation"""
        save_path = Path(temp_dir) / "pareto_2d.png"
        plot_pareto_front_2d(pareto_2d, save_path)
        assert save_path.exists()

    def test_2d_plot_with_knee_point(self, temp_dir, pareto_2d):
        """Test 2D plot with knee point highlighting"""
        save_path = Path(temp_dir) / "pareto_2d_knee.png"
        plot_pareto_front_2d(pareto_2d, save_path, show_knee_point=True)
        assert save_path.exists()

    def test_2d_plot_custom_labels(self, temp_dir, pareto_2d):
        """Test 2D plot with custom axis labels"""
        save_path = Path(temp_dir) / "pareto_2d_labels.png"
        plot_pareto_front_2d(
            pareto_2d, save_path, xlabel="Dose Error (%)", ylabel="Efficiency Loss (%)"
        )
        assert save_path.exists()

    def test_2d_plot_custom_title(self, temp_dir, pareto_2d):
        """Test 2D plot with custom title"""
        save_path = Path(temp_dir) / "pareto_2d_title.png"
        plot_pareto_front_2d(pareto_2d, save_path, title="Custom Pareto Front Title")
        assert save_path.exists()

    def test_2d_plot_custom_dpi(self, temp_dir, pareto_2d):
        """Test 2D plot with custom DPI"""
        save_path = Path(temp_dir) / "pareto_2d_dpi.png"
        plot_pareto_front_2d(pareto_2d, save_path, dpi=150)
        assert save_path.exists()

    def test_2d_plot_single_point(self, temp_dir):
        """Test 2D plot with single solution"""
        single_point = np.array([[1.0, 2.0]])
        save_path = Path(temp_dir) / "pareto_2d_single.png"
        plot_pareto_front_2d(single_point, save_path)
        assert save_path.exists()

    def test_2d_plot_many_points(self, temp_dir):
        """Test 2D plot with many solutions"""
        np.random.seed(42)
        many_points = np.random.rand(100, 2) * 10
        save_path = Path(temp_dir) / "pareto_2d_many.png"
        plot_pareto_front_2d(many_points, save_path)
        assert save_path.exists()


# ============================================================================
# 3D Pareto Front Tests
# ============================================================================


class TestParetoFront3D:
    """Test 3D Pareto front plotting"""

    def test_basic_3d_plot(self, temp_dir, pareto_3d):
        """Test basic 3D Pareto front plot creation"""
        save_path = Path(temp_dir) / "pareto_3d.png"
        plot_pareto_front_3d(pareto_3d, save_path)
        assert save_path.exists()

    def test_3d_plot_with_knee_point(self, temp_dir, pareto_3d):
        """Test 3D plot with knee point highlighting"""
        save_path = Path(temp_dir) / "pareto_3d_knee.png"
        plot_pareto_front_3d(pareto_3d, save_path, show_knee_point=True)
        assert save_path.exists()

    def test_3d_plot_custom_labels(self, temp_dir, pareto_3d):
        """Test 3D plot with custom axis labels"""
        save_path = Path(temp_dir) / "pareto_3d_labels.png"
        plot_pareto_front_3d(pareto_3d, save_path, labels=["Obj 1", "Obj 2", "Obj 3"])
        assert save_path.exists()

    def test_3d_plot_custom_title(self, temp_dir, pareto_3d):
        """Test 3D plot with custom title"""
        save_path = Path(temp_dir) / "pareto_3d_title.png"
        plot_pareto_front_3d(pareto_3d, save_path, title="Custom 3D Title")
        assert save_path.exists()

    def test_3d_plot_single_point(self, temp_dir):
        """Test 3D plot with single solution"""
        single_point = np.array([[1.0, 2.0, 3.0]])
        save_path = Path(temp_dir) / "pareto_3d_single.png"
        plot_pareto_front_3d(single_point, save_path)
        assert save_path.exists()


# ============================================================================
# High-Dimensional (Projections) Tests
# ============================================================================


class TestParetoFrontProjections:
    """Test high-dimensional Pareto front projection plotting"""

    def test_4d_projections(self, temp_dir, pareto_4d):
        """Test 4D Pareto front pairwise projections"""
        save_path = Path(temp_dir) / "pareto_4d.png"
        plot_pareto_front_projections(pareto_4d, save_path)
        assert save_path.exists()

    def test_5d_projections(self, temp_dir, pareto_5d):
        """Test 5D Pareto front pairwise projections"""
        save_path = Path(temp_dir) / "pareto_5d.png"
        plot_pareto_front_projections(pareto_5d, save_path)
        assert save_path.exists()

    def test_projections_custom_title(self, temp_dir, pareto_4d):
        """Test projections with custom title"""
        save_path = Path(temp_dir) / "pareto_4d_title.png"
        plot_pareto_front_projections(
            pareto_4d, save_path, title="Custom Projection Title"
        )
        assert save_path.exists()


# ============================================================================
# Parallel Coordinates Tests
# ============================================================================


class TestParallelCoordinates:
    """Test parallel coordinates plotting"""

    def test_basic_parallel_coords(self, temp_dir, pareto_2d):
        """Test basic parallel coordinates plot"""
        save_path = Path(temp_dir) / "parallel_2d.png"
        plot_parallel_coordinates(pareto_2d, save_path)
        assert save_path.exists()

    def test_parallel_coords_3d(self, temp_dir, pareto_3d):
        """Test parallel coordinates with 3 objectives"""
        save_path = Path(temp_dir) / "parallel_3d.png"
        plot_parallel_coordinates(pareto_3d, save_path)
        assert save_path.exists()

    def test_parallel_coords_5d(self, temp_dir, pareto_5d):
        """Test parallel coordinates with 5 objectives"""
        save_path = Path(temp_dir) / "parallel_5d.png"
        plot_parallel_coordinates(pareto_5d, save_path)
        assert save_path.exists()

    def test_parallel_coords_custom_labels(self, temp_dir, pareto_3d):
        """Test parallel coordinates with custom labels"""
        save_path = Path(temp_dir) / "parallel_labels.png"
        plot_parallel_coordinates(
            pareto_3d, save_path, objective_names=["Dose", "Efficiency", "Conformity"]
        )
        assert save_path.exists()

    def test_parallel_coords_highlight(self, temp_dir, pareto_3d):
        """Test parallel coordinates with highlighted solutions"""
        save_path = Path(temp_dir) / "parallel_highlight.png"
        plot_parallel_coordinates(pareto_3d, save_path, highlight_solutions=[0, 2])
        assert save_path.exists()

    def test_parallel_coords_with_rank(self, temp_dir, pareto_3d):
        """Test parallel coordinates with dominance rank coloring"""
        save_path = Path(temp_dir) / "parallel_rank.png"
        ranks = calculate_dominance_rank(pareto_3d)
        plot_parallel_coordinates(pareto_3d, save_path, dominance_rank=ranks)
        assert save_path.exists()


# ============================================================================
# Petal Diagram Tests
# ============================================================================


class TestPetalDiagrams:
    """Test petal (radar) diagram plotting"""

    def test_single_petal_3obj(self, temp_dir, pareto_3d):
        """Test single solution petal diagram with 3 objectives"""
        save_path = Path(temp_dir) / "petal_single_3d.png"
        solution = pareto_3d[0]
        plot_petal_diagram_single(solution, save_path)
        assert save_path.exists()

    def test_single_petal_5obj(self, temp_dir, pareto_5d):
        """Test single solution petal diagram with 5 objectives"""
        save_path = Path(temp_dir) / "petal_single_5d.png"
        solution = pareto_5d[0]
        plot_petal_diagram_single(solution, save_path)
        assert save_path.exists()

    def test_single_petal_custom_labels(self, temp_dir, pareto_3d):
        """Test single petal with custom labels"""
        save_path = Path(temp_dir) / "petal_labels.png"
        solution = pareto_3d[0]
        plot_petal_diagram_single(
            solution, save_path, objective_names=["Dose", "Efficiency", "Conformity"]
        )
        assert save_path.exists()

    def test_multi_petal(self, temp_dir, pareto_3d):
        """Test multiple solution petal diagrams"""
        save_dir = Path(temp_dir) / "petal_multi"
        plot_petal_diagram_multi(pareto_3d, save_dir, max_solutions=3)
        assert save_dir.exists()
        # Should have created individual petal diagrams
        assert len(list(save_dir.glob("*.png"))) > 0


# ============================================================================
# Metric Calculation Tests
# ============================================================================


class TestKneePointCalculation:
    """Test knee point calculation"""

    def test_knee_point_2d(self, pareto_2d):
        """Test knee point calculation for 2D data"""
        knee_idx = calculate_knee_point(pareto_2d)
        assert isinstance(knee_idx, (int, np.integer))
        assert 0 <= knee_idx < len(pareto_2d)

    def test_knee_point_3d(self, pareto_3d):
        """Test knee point calculation for 3D data"""
        knee_idx = calculate_knee_point(pareto_3d)
        assert isinstance(knee_idx, (int, np.integer))
        assert 0 <= knee_idx < len(pareto_3d)

    def test_knee_point_single_solution(self):
        """Test knee point with single solution returns index 0"""
        single = np.array([[1.0, 2.0]])
        knee_idx = calculate_knee_point(single)
        assert knee_idx == 0

    def test_knee_point_returns_valid_index(self, pareto_2d):
        """Test that knee point returns a valid index"""
        knee_idx = calculate_knee_point(pareto_2d)
        # Should be able to use it to index the array
        knee_solution = pareto_2d[knee_idx]
        assert len(knee_solution) == 2


class TestCrowdingDistance:
    """Test crowding distance calculation"""

    def test_crowding_distance_2d(self, pareto_2d):
        """Test crowding distance for 2D data"""
        distances = calculate_crowding_distance(pareto_2d)
        assert len(distances) == len(pareto_2d)

    def test_crowding_distance_3d(self, pareto_3d):
        """Test crowding distance for 3D data"""
        distances = calculate_crowding_distance(pareto_3d)
        assert len(distances) == len(pareto_3d)

    def test_crowding_distance_boundary_infinite(self, pareto_2d):
        """Test that boundary solutions have infinite crowding distance"""
        distances = calculate_crowding_distance(pareto_2d)
        # After sorting, boundary points should have infinite distance
        assert np.any(np.isinf(distances))

    def test_crowding_distance_single_solution(self):
        """Test crowding distance with single solution"""
        single = np.array([[1.0, 2.0]])
        distances = calculate_crowding_distance(single)
        assert len(distances) == 1
        assert np.isinf(distances[0])

    def test_crowding_distance_two_solutions(self):
        """Test crowding distance with two solutions"""
        two_points = np.array([[1.0, 2.0], [3.0, 1.0]])
        distances = calculate_crowding_distance(two_points)
        assert len(distances) == 2
        # Both should be infinite (boundaries)
        assert np.all(np.isinf(distances))


class TestDominanceRank:
    """Test dominance rank calculation"""

    def test_dominance_rank_simple(self):
        """Test dominance rank with clear hierarchy"""
        objectives = np.array(
            [
                [1.0, 1.0],  # Rank 0 (dominates all)
                [2.0, 2.0],  # Rank 1
                [3.0, 3.0],  # Rank 2
            ]
        )
        ranks = calculate_dominance_rank(objectives)
        assert ranks[0] == 0
        assert ranks[1] == 1
        assert ranks[2] == 2

    def test_dominance_rank_pareto_front(self, pareto_2d):
        """Test dominance rank for actual Pareto front"""
        # If this is actually a Pareto front, all should be rank 0
        # Our test data may have mixed dominance
        ranks = calculate_dominance_rank(pareto_2d)
        assert len(ranks) == len(pareto_2d)
        assert np.min(ranks) == 0  # At least some should be non-dominated

    def test_dominance_rank_all_non_dominated(self):
        """Test when all solutions are non-dominated"""
        objectives = np.array(
            [
                [1.0, 4.0],
                [2.0, 3.0],
                [3.0, 2.0],
                [4.0, 1.0],
            ]
        )
        ranks = calculate_dominance_rank(objectives)
        # All should be rank 0
        np.testing.assert_array_equal(ranks, np.zeros(4))

    def test_dominance_rank_single_solution(self):
        """Test dominance rank with single solution"""
        single = np.array([[1.0, 2.0]])
        ranks = calculate_dominance_rank(single)
        assert len(ranks) == 1
        assert ranks[0] == 0


# ============================================================================
# Log File I/O Tests
# ============================================================================


class TestLogFileReading:
    """Test log file reading functionality"""

    def test_read_log_file_basic(self, sample_log_file):
        """Test basic log file reading"""
        data = ReadInMultiObjectiveLogFile(sample_log_file)

        assert "Iteration" in data
        assert "param1" in data
        assert "param2" in data
        assert "ObjectiveFunction_1" in data
        assert "ObjectiveFunction_2" in data

    def test_read_log_file_correct_values(self, sample_log_file):
        """Test that values are correctly parsed"""
        data = ReadInMultiObjectiveLogFile(sample_log_file)

        assert data["Iteration"][0] == 0
        assert data["param1"][0] == 5.0
        assert data["ObjectiveFunction_1"][0] == 10.0

    def test_read_log_file_3_objectives(self, sample_log_file_3obj):
        """Test reading log file with 3 objectives"""
        data = ReadInMultiObjectiveLogFile(sample_log_file_3obj)

        assert "ObjectiveFunction_3" in data
        assert len(data["ObjectiveFunction_3"]) == 3


class TestLogFileWriting:
    """Test log file writing functionality"""

    def test_log_pareto_front(self, temp_dir, pareto_2d):
        """Test logging Pareto front to file"""
        log_path = Path(temp_dir) / "pareto_log.txt"
        parameter_names = ["param1", "param2"]

        LogParetoFrontToFile(log_path, pareto_2d, parameter_names, n_objectives=2)

        assert log_path.exists()

        # Check content
        with open(log_path, "r") as f:
            lines = f.readlines()
            assert "Solution_Index" in lines[0]
            assert "Objective_1" in lines[0]
            assert len(lines) == 6  # Header + 5 solutions

    def test_log_pareto_front_3d(self, temp_dir, pareto_3d):
        """Test logging 3D Pareto front"""
        log_path = Path(temp_dir) / "pareto_log_3d.txt"
        parameter_names = ["x1", "x2", "x3"]

        LogParetoFrontToFile(log_path, pareto_3d, parameter_names, n_objectives=3)

        assert log_path.exists()

        with open(log_path, "r") as f:
            lines = f.readlines()
            assert "Objective_3" in lines[0]


# ============================================================================
# Convergence Plotting Tests
# ============================================================================


class TestConvergencePlotting:
    """Test convergence plotting functions"""

    def test_objective_convergence(self, temp_dir, sample_log_file):
        """Test objective convergence plotting"""
        save_path = Path(temp_dir) / "obj_convergence.png"
        plot_objective_convergence(sample_log_file, save_path, n_objectives=2)
        assert save_path.exists()

    def test_objective_convergence_3obj(self, temp_dir, sample_log_file_3obj):
        """Test objective convergence with 3 objectives"""
        save_path = Path(temp_dir) / "obj_convergence_3d.png"
        plot_objective_convergence(sample_log_file_3obj, save_path, n_objectives=3)
        assert save_path.exists()

    def test_parameter_convergence(self, temp_dir, sample_log_file):
        """Test parameter convergence plotting"""
        save_path = Path(temp_dir) / "param_convergence.png"
        plot_parameter_convergence(sample_log_file, save_path, ["param1", "param2"])
        assert save_path.exists()


# ============================================================================
# Style and Configuration Tests
# ============================================================================


class TestStyleSetup:
    """Test style configuration functions"""

    def test_apply_default_style(self):
        """Default apply_style call should not raise."""
        apply_style()

    def test_apply_fast_style(self):
        """Explicit fast style setup runs without error."""
        apply_style("fast")

    def test_apply_publication_style(self):
        """Explicit publication style setup runs without error."""
        apply_style("publication")

    def test_invalid_style_raises(self):
        """Only fast/publication styles should be accepted."""
        with pytest.raises(ValueError):
            apply_style("journal")

    def test_saving_publication_plot_does_not_emit_tight_layout_warning(
        self, temp_dir, pareto_2d
    ):
        """Saving should stay quiet even when constrained layout is active."""
        apply_style("publication", variant="clean")
        save_path = Path(temp_dir) / "quiet_pareto.png"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plot_pareto_front_2d(pareto_2d, save_path)

        warning_text = "\n".join(str(item.message) for item in caught)
        assert "figure layout has changed to tight" not in warning_text

    def test_medicalphysics_style_uses_single_column_size(self):
        """Medical Physics variant authors at 2x the 80 mm single column.

        The journal scales the figure down to the 80 mm column, which turns the
        mandated >=20 pt fonts into ~10 pt legible print text. We therefore
        author at twice the column width rather than at 80 mm directly.
        """
        from TopasMOO.plotting.style import MEDICAL_PHYSICS_SINGLE_COL_WIDTH

        apply_style("publication", variant="medicalphysics")
        assert matplotlib.rcParams["savefig.dpi"] == 600
        assert matplotlib.rcParams["figure.figsize"][0] == pytest.approx(
            MEDICAL_PHYSICS_SINGLE_COL_WIDTH
        )
        assert matplotlib.rcParams["font.size"] >= 20
        apply_style("fast")


# ============================================================================
# Parameter-Objective Correlation Tests
# ============================================================================


class TestParameterObjectiveCorrelation:
    """Test parameter-objective correlation plotting"""

    def test_basic_correlation_plot(self, temp_dir, decision_vars, pareto_2d):
        """Test basic correlation plot"""
        save_path = Path(temp_dir) / "correlation.png"
        plot_parameter_objective_correlation(decision_vars, pareto_2d, save_path)
        assert save_path.exists()


class TestDecisionHeatmap:
    """Test decision heatmap layout."""

    def test_mean_median_legend_sits_right_of_boxplot(self, decision_vars):
        _, box_ax = plot_decision_heatmap(
            decision_vars, parameter_names=["x1", "x2"]
        )
        figure = box_ax.figure

        try:
            figure.canvas.draw()
            legend_bounds = box_ax.get_legend().get_window_extent()
            axes_bounds = box_ax.get_window_extent()

            assert legend_bounds.x0 >= axes_bounds.x1
            assert legend_bounds.x1 <= figure.bbox.x1
            assert len(figure.axes) == 3  # heatmap, boxplot, and colorbar
        finally:
            plt.close(figure)

    def test_decision_heatmap_has_publication_sized_output(self, temp_dir, decision_vars):
        save_path = Path(temp_dir) / "decision_heatmap.png"
        plot_decision_heatmap(decision_vars, save_path, parameter_names=["x1", "x2"])

        img = mpimg.imread(save_path)
        height, width = img.shape[:2]
        assert width >= 1800
        assert height >= 900

    def test_correlation_with_names(self, temp_dir, decision_vars, pareto_2d):
        """Test correlation plot with custom names"""
        save_path = Path(temp_dir) / "correlation_names.png"
        plot_parameter_objective_correlation(
            decision_vars,
            pareto_2d,
            save_path,
            parameter_names=["x", "y"],
            objective_names=["f1", "f2"],
        )
        assert save_path.exists()


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_array_handling(self, temp_dir):
        """Test handling of empty arrays"""
        empty = np.array([]).reshape(0, 2)
        save_path = Path(temp_dir) / "empty.png"
        # Should handle gracefully (may skip plotting)
        try:
            plot_pareto_front_2d(empty, save_path)
        except (ValueError, IndexError):
            pass  # Expected for empty arrays

    def test_very_small_values(self, temp_dir):
        """Test with very small objective values"""
        small_values = np.array([[1e-10, 2e-10], [3e-10, 1e-10]])
        save_path = Path(temp_dir) / "small_values.png"
        plot_pareto_front_2d(small_values, save_path)
        assert save_path.exists()

    def test_very_large_values(self, temp_dir):
        """Test with very large objective values"""
        large_values = np.array([[1e10, 2e10], [3e10, 1e10]])
        save_path = Path(temp_dir) / "large_values.png"
        plot_pareto_front_2d(large_values, save_path)
        assert save_path.exists()

    def test_negative_values(self, temp_dir):
        """Test with negative objective values"""
        negative = np.array([[-1.0, -2.0], [-3.0, -1.0]])
        save_path = Path(temp_dir) / "negative.png"
        plot_pareto_front_2d(negative, save_path)
        assert save_path.exists()

    def test_mixed_values(self, temp_dir):
        """Test with mixed positive/negative values"""
        mixed = np.array([[-1.0, 2.0], [3.0, -1.0], [0.0, 0.0]])
        save_path = Path(temp_dir) / "mixed.png"
        plot_pareto_front_2d(mixed, save_path)
        assert save_path.exists()

    def test_identical_solutions(self, temp_dir):
        """Test with identical solutions"""
        identical = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        save_path = Path(temp_dir) / "identical.png"
        plot_pareto_front_2d(identical, save_path)
        assert save_path.exists()


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_full_workflow_2d(
        self, temp_dir, pareto_2d, decision_vars, sample_log_file
    ):
        """Test complete 2D visualization workflow"""
        # Create all 2D visualizations
        save_dir = Path(temp_dir) / "full_workflow"
        save_dir.mkdir()

        # Pareto front
        plot_pareto_front_2d(pareto_2d, save_dir / "pareto.png")

        # Parallel coordinates
        plot_parallel_coordinates(pareto_2d, save_dir / "parallel.png")

        # Convergence
        plot_objective_convergence(
            sample_log_file, save_dir / "convergence.png", n_objectives=2
        )
        plot_parameter_convergence(
            sample_log_file, save_dir / "param_conv.png", ["param1", "param2"]
        )

        # Log Pareto front
        LogParetoFrontToFile(
            save_dir / "pareto.txt", pareto_2d, ["p1", "p2"], n_objectives=2
        )

        # Verify all files created
        assert (save_dir / "pareto.png").exists()
        assert (save_dir / "parallel.png").exists()
        assert (save_dir / "convergence.png").exists()
        assert (save_dir / "param_conv.png").exists()
        assert (save_dir / "pareto.txt").exists()

    def test_full_workflow_3d(self, temp_dir, pareto_3d):
        """Test complete 3D visualization workflow"""
        save_dir = Path(temp_dir) / "full_workflow_3d"
        save_dir.mkdir()

        # 3D Pareto front
        plot_pareto_front_3d(pareto_3d, save_dir / "pareto_3d.png")

        # Parallel coordinates
        plot_parallel_coordinates(pareto_3d, save_dir / "parallel_3d.png")

        # Petal diagrams
        plot_petal_diagram_multi(pareto_3d, save_dir / "petals", max_solutions=2)

        # Verify files created
        assert (save_dir / "pareto_3d.png").exists()
        assert (save_dir / "parallel_3d.png").exists()
        assert (save_dir / "petals").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
