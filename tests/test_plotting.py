"""Plot content, representative exports, layout regressions, metrics, and log I/O."""

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
def pareto_5d():
    """Synthetic 5D Pareto front data"""
    return np.random.default_rng(123).random((20, 5)) * 10


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


class TestParetoFront2D:
    def test_data_labels_knee_and_export(self, tmp_path, pareto_2d):
        from PIL import Image

        path = tmp_path / "pareto.png"
        ax = plot_pareto_front_2d(
            pareto_2d,
            path,
            true_front=pareto_2d,
            show_knee_point=True,
            xlabel="Dose Error (%)",
            ylabel="Efficiency Loss (%)",
            title="Trade-offs",
            dpi=150,
        )
        assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == (
            "Dose Error (%)",
            "Efficiency Loss (%)",
            "Trade-offs",
        )
        np.testing.assert_array_equal(ax.collections[0].get_offsets(), pareto_2d)
        np.testing.assert_array_equal(ax.collections[1].get_offsets(), pareto_2d[[1]])
        np.testing.assert_array_equal(ax.lines[0].get_xydata(), pareto_2d)
        assert [t.get_text() for t in ax.figure.legends[0].get_texts()] == [
            "True Pareto Front",
            "Obtained Solutions",
            "Knee point",
        ]
        assert path.with_suffix(".pdf").is_file()
        with Image.open(path) as saved:
            # PNG stores integer pixels/metre, so DPI is rounded on conversion.
            assert saved.info["dpi"] == pytest.approx((150, 150), abs=0.02)

    def test_single_point(self):
        ax = plot_pareto_front_2d(np.array([[1.0, 2.0]]), show_knee_point=True)
        try:
            np.testing.assert_array_equal(ax.collections[1].get_offsets(), [[1, 2]])
            assert np.isfinite([ax.get_xlim(), ax.get_ylim()]).all()
        finally:
            plt.close(ax.figure)


class TestParetoFront3D:
    def test_data_labels_knee_and_export(self, tmp_path, pareto_3d):
        path = tmp_path / "pareto_3d.png"
        ax = plot_pareto_front_3d(
            pareto_3d,
            path,
            show_knee_point=True,
            labels=["Dose", "Time", "Error"],
            title="Three objectives",
        )
        assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_zlabel()) == ("Dose", "Time", "Error")
        assert ax.get_title() == "Three objectives"
        np.testing.assert_array_equal(np.column_stack(ax.collections[0]._offsets3d), pareto_3d)
        np.testing.assert_array_equal(np.column_stack(ax.collections[1]._offsets3d), pareto_3d[[4]])
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Knee point"]
        assert path.is_file() and path.with_suffix(".pdf").is_file()

    def test_single_point(self):
        ax = plot_pareto_front_3d(np.array([[1.0, 2.0, 3.0]]), show_knee_point=True)
        try:
            np.testing.assert_array_equal(
                np.column_stack(ax.collections[1]._offsets3d), [[1, 2, 3]]
            )
        finally:
            plt.close(ax.figure)


class TestParetoFrontProjections:
    def test_projection_pairs_and_export(self, tmp_path, pareto_5d):
        from itertools import combinations

        path = tmp_path / "projections.png"
        names = ["A", "B", "C", "D", "E"]
        axes = plot_pareto_front_projections(
            pareto_5d, path, objective_names=names, title="Objective pairs"
        )
        for ax, (i, j) in zip(axes, combinations(range(5), 2)):
            assert (ax.get_xlabel(), ax.get_ylabel()) == (names[i], names[j])
            np.testing.assert_array_equal(ax.collections[0].get_offsets(), pareto_5d[:, [i, j]])
        assert len(axes) == 12
        assert all(not ax.axison for ax in axes[10:])
        assert axes[0].figure.get_suptitle() == "Objective pairs"
        assert path.is_file() and path.with_suffix(".pdf").is_file()


class TestParallelCoordinates:
    def test_normalization_labels_highlights_and_export(self, tmp_path):
        front = np.array([[0, 10, 5], [1, 5, 5], [2, 0, 5]])
        path = tmp_path / "parallel.png"
        ax = plot_parallel_coordinates(
            front,
            path,
            objective_names=["Dose", "Time", "Error"],
            highlight_solutions=[0, 2],
        )
        assert [t.get_text() for t in ax.get_xticklabels()] == ["Dose", "Time", "Error"]
        np.testing.assert_allclose(
            [line.get_ydata() for line in ax.lines[3:6]],
            [[0, 1, 0], [0.5, 0.5, 0], [1, 0, 0]],
        )
        highlighted = [line for line in ax.lines if line.get_marker() == "o"]
        np.testing.assert_array_equal(
            [line.get_ydata() for line in highlighted], [[0, 1, 0], [1, 0, 0]]
        )
        assert path.is_file() and path.with_suffix(".pdf").is_file()

    def test_rank_coloring(self, pareto_3d):
        ax = plot_parallel_coordinates(pareto_3d, dominance_rank=np.arange(5))
        try:
            colors = [line.get_color() for line in ax.lines[3:]]
            assert len(set(colors)) == 5
            assert ax.figure.axes[1].get_ylabel() == "Dominance Rank"
            assert ax.get_legend() is None
        finally:
            plt.close(ax.figure)


class TestPetalDiagrams:
    def test_single_petal_values_labels_and_export(self, tmp_path):
        path = tmp_path / "petal.png"
        ax = plot_petal_diagram_single(
            np.array([1.0, 2.0, 4.0]), path, objective_names=["Dose", "Time", "Error"]
        )
        assert [t.get_text() for t in ax.get_xticklabels()] == ["Dose", "Time", "Error"]
        np.testing.assert_allclose([bar.get_height() for bar in ax.patches], [0.75, 0.5, 0.03])
        assert path.is_file() and path.with_suffix(".pdf").is_file()

    def test_multi_petal(self, tmp_path, pareto_3d):
        fig = plot_petal_diagram_multi(pareto_3d, tmp_path, max_solutions=3)
        assert len(fig.axes) == 3
        assert {p.stem for p in tmp_path.glob("*.png")} == {
            "petal_solution_1",
            "petal_solution_3",
            "petal_solution_5",
            "petal_diagram_multipanel",
        }
        assert {p.stem for p in tmp_path.glob("*.pdf")} == {p.stem for p in tmp_path.glob("*.png")}


class TestKneePointCalculation:
    def test_knee_uses_normalized_tradeoff(self):
        # The middle point minimizes the normalized sum, even with unequal units
        # and a constant objective. A raw sum would incorrectly pick row zero.
        front = np.array([[0, 10, 7], [200, 2, 7], [1000, 0, 7]])
        assert calculate_knee_point(front) == 1

    def test_single_solution(self):
        assert calculate_knee_point(np.array([[1.0, 2.0]])) == 0


class TestCrowdingDistance:
    def test_interior_distances_and_boundaries(self):
        front = np.array([[2, 2], [0, 4], [4, 0], [1, 3]], dtype=float)
        np.testing.assert_allclose(calculate_crowding_distance(front), [1.5, np.inf, np.inf, 1.0])

    @pytest.mark.parametrize("front", [[[1.0, 2.0]], [[1.0, 2.0], [3.0, 1.0]]])
    def test_small_fronts_are_all_boundaries(self, front):
        np.testing.assert_array_equal(
            calculate_crowding_distance(np.array(front)), [np.inf] * len(front)
        )


class TestDominanceRank:
    def test_dominance_layers_and_ties(self):
        front = np.array([[2, 2], [1, 3], [1, 1], [3, 3], [1, 1]])
        np.testing.assert_array_equal(calculate_dominance_rank(front), [1, 1, 0, 2, 0])

    def test_all_non_dominated(self, pareto_2d):
        np.testing.assert_array_equal(calculate_dominance_rank(pareto_2d), np.zeros(5))


class TestLogFileReading:
    def test_read_log_file_3_objectives(self, sample_log_file_3obj):
        data = ReadInMultiObjectiveLogFile(sample_log_file_3obj)
        assert data["ObjectiveFunction_3"] == [12.0, 11.0, 10.0]


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


class TestConvergencePlotting:
    def test_objective_convergence(self, tmp_path, sample_log_file_3obj):
        path = tmp_path / "convergence.png"
        axes = plot_objective_convergence(sample_log_file_3obj, path, n_objectives=3)
        assert axes.shape == (2, 2)
        assert not axes[1, 1].get_visible()
        for ax, expected in zip(axes.flat, [[10, 9, 8], [15, 14, 13], [12, 11, 10]]):
            np.testing.assert_array_equal(ax.lines[0].get_ydata(), expected)
        assert path.is_file() and path.with_suffix(".pdf").is_file()

    def test_parameter_convergence(self, tmp_path, sample_log_file):
        path = tmp_path / "parameters.png"
        axes = plot_parameter_convergence(sample_log_file, path, ["param1", "param2"])
        assert [ax.get_ylabel() for ax in axes] == ["param1", "param2"]
        np.testing.assert_array_equal(
            axes[0].collections[0].get_offsets()[:, 1], [5, 4.5, 4, 3.5, 3]
        )
        np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), [4.5, 4.25, 4, 3.75, 3.5])
        assert path.is_file() and path.with_suffix(".pdf").is_file()


class TestStyleSetup:
    """Test style configuration functions"""

    def test_invalid_style_raises(self):
        """Only fast/publication styles should be accepted."""
        with pytest.raises(ValueError):
            apply_style("journal")

    def test_saving_publication_plot_does_not_emit_tight_layout_warning(self, temp_dir, pareto_2d):
        """Saving should stay quiet even when constrained layout is active."""
        apply_style("publication", variant="clean")
        save_path = Path(temp_dir) / "quiet_pareto.png"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plot_pareto_front_2d(pareto_2d, save_path)

        warning_text = "\n".join(str(item.message) for item in caught)
        assert "figure layout has changed to tight" not in warning_text

    def test_medicalphysics_style_uses_single_column_size(self):
        """Medical Physics variant authors at the 80 mm single column.

        The journal does not rescale typeset figures, so the 10 pt fonts print
        at 10 pt.
        """
        from TopasMOO.plotting.style import MEDICAL_PHYSICS_SINGLE_COL_WIDTH

        apply_style("publication", variant="medicalphysics")
        assert matplotlib.rcParams["savefig.dpi"] == 600
        assert matplotlib.rcParams["figure.figsize"][0] == pytest.approx(
            MEDICAL_PHYSICS_SINGLE_COL_WIDTH
        )
        assert matplotlib.rcParams["font.size"] == 10
        apply_style("fast")

    def test_medicalphysics_saves_at_exact_column_width(self, temp_dir):
        """The saved PNG is exactly 80 mm wide, since the journal won't rescale."""
        front = np.column_stack([np.linspace(0, 1, 5), np.linspace(1, 0, 5)])
        save_path = Path(temp_dir) / "medphys.png"
        try:
            apply_style("publication", variant="medicalphysics")
            plot_pareto_front_2d(front, save_path, show_knee_point=True)
        finally:
            apply_style("fast")
        width_px = mpimg.imread(save_path).shape[1]
        assert width_px / 600 * 25.4 == pytest.approx(80, abs=0.1)

    @pytest.mark.parametrize("show_knee_point", [False, True])
    def test_medicalphysics_markers_and_legend_follow_style(self, show_knee_point):
        """Scatter edges and legend frame come from the style, not hardcoded."""
        from TopasMOO.plotting.style import marker_edge_width

        front = np.column_stack([np.linspace(0, 1, 5), np.linspace(1, 0, 5)])
        try:
            apply_style("publication", variant="medicalphysics")
            assert marker_edge_width(0.25) == 0
            ax = plot_pareto_front_2d(front, true_front=front, show_knee_point=show_knee_point)
            (legend,) = ax.figure.legends
            assert not legend.get_frame_on()
            ax.figure.canvas.draw()
            renderer = ax.figure.canvas.get_renderer()
            legend_box = legend.get_window_extent(renderer)
            assert legend_box.y1 <= ax.get_tightbbox(renderer).y0
            assert legend_box.x0 >= ax.figure.bbox.x0
            assert legend_box.x1 <= ax.figure.bbox.x1
            assert all((c.get_linewidths() == 0).all() for c in ax.collections)
            plt.close(ax.figure)

            apply_style("publication", variant="clean")
            assert marker_edge_width(0.25) > 0
        finally:
            apply_style("fast")


class TestParameterObjectiveCorrelation:
    def test_correlation_data_labels_and_export(self, tmp_path, decision_vars, pareto_2d):
        path = tmp_path / "correlation.png"
        axes = plot_parameter_objective_correlation(
            decision_vars,
            pareto_2d,
            path,
            parameter_names=["x", "y"],
            objective_names=["f1", "f2"],
        )
        assert axes.shape == (2, 2)
        assert [ax.get_xlabel() for ax in axes[-1]] == ["x", "y"]
        assert [ax.get_ylabel() for ax in axes[:, 0]] == ["f1", "f2"]
        for i, j in np.ndindex(2, 2):
            np.testing.assert_array_equal(
                axes[i, j].collections[0].get_offsets(),
                np.column_stack([decision_vars[:, j], pareto_2d[:, i]]),
            )
        assert path.is_file() and path.with_suffix(".pdf").is_file()


class TestDecisionHeatmap:
    """Test decision heatmap layout."""

    def test_mean_median_legend_sits_right_of_boxplot(self, decision_vars):
        _, box_ax = plot_decision_heatmap(decision_vars, parameter_names=["x1", "x2"])
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


class TestEdgeCases:
    def test_empty_front_has_no_points(self):
        ax = plot_pareto_front_2d(np.empty((0, 2)))
        try:
            ax.figure.canvas.draw()
            assert ax.collections[0].get_offsets().shape == (0, 2)
        finally:
            plt.close(ax.figure)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
