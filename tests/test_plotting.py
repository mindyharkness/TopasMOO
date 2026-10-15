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
    hypervolume_curve,
    plot_constraint_check,
    plot_decision_along_front,
    plot_hypervolume_vs_evaluations,
    plot_parallel_coordinates,
    plot_parameter_convergence,
    plot_parameter_objective_correlation,
    plot_pareto_front,
    plot_pareto_front_2d,
    plot_pareto_front_projections,
    plot_run_monitor,
    plot_solution_table,
)
from TopasMOO.plotting.style import apply_style
from TopasMOO.plotting.table import select_solutions

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
    def test_front_knee_labels_and_export(self, tmp_path, pareto_2d):
        from PIL import Image

        path = tmp_path / "pareto.png"
        ax = plot_pareto_front_2d(
            pareto_2d,
            path,
            true_front=pareto_2d,
            objective_names=["Dose Error (%)", "Efficiency Loss (%)"],
            dpi=150,
        )
        assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == (
            "Dose Error (%)",
            "Efficiency Loss (%)",
            "",
        )
        np.testing.assert_array_equal(ax.collections[0].get_offsets(), pareto_2d)
        np.testing.assert_array_equal(ax.collections[1].get_offsets(), pareto_2d[[1]])
        assert {t.get_text() for t in ax.get_legend().get_texts()} == {
            "True front",
            "Pareto front",
            "Knee point",
        }
        assert path.with_suffix(".pdf").is_file()
        with Image.open(path) as saved:
            # PNG stores integer pixels/meter, so DPI is rounded on conversion.
            assert saved.info["dpi"] == pytest.approx((150, 150), abs=0.02)

    def test_context_adds_gray_designs_and_crops_far_infeasible(self, pareto_2d):
        evaluated = np.vstack([pareto_2d, [[6.0, 6.0], [100.0, 100.0]]])
        eligible = np.r_[np.ones(6, bool), False]
        ax = plot_pareto_front_2d(pareto_2d, evaluated_objectives=evaluated, eligible=eligible)
        try:
            assert {t.get_text() for t in ax.get_legend().get_texts()} == {
                "Other designs",
                "Infeasible",
                "Pareto front",
                "Knee point",
            }
            assert ax.get_xlim()[1] < 100  # framed by the feasible designs
        finally:
            plt.close(ax.figure)

    def test_maximize_flips_display_not_selection(self, pareto_2d):
        ax = plot_pareto_front_2d(pareto_2d, maximize=[False, True])
        try:
            np.testing.assert_array_equal(ax.collections[0].get_offsets()[:, 1], -pareto_2d[:, 1])
            np.testing.assert_array_equal(ax.collections[1].get_offsets(), [[2.0, -3.0]])
        finally:
            plt.close(ax.figure)

    def test_single_point(self):
        ax = plot_pareto_front_2d(np.array([[1.0, 2.0]]))
        try:
            np.testing.assert_array_equal(ax.collections[1].get_offsets(), [[1, 2]])
            assert np.isfinite([ax.get_xlim(), ax.get_ylim()]).all()
        finally:
            plt.close(ax.figure)


class TestParetoFrontDispatcherAndProjections:
    def test_dispatches_by_objective_count(self, pareto_2d, pareto_3d):
        ax = plot_pareto_front(pareto_2d, objective_names=["Dose (Gy)", "Time (s)"])
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("Dose (Gy)", "Time (s)")
        assert len(plot_pareto_front(pareto_3d)) == 3
        with pytest.raises(ValueError, match="objective_names"):
            plot_pareto_front(pareto_2d, objective_names=["only one"])
        plt.close("all")

    def test_pairs_colored_by_remaining_objective(self, tmp_path, pareto_3d):
        axes = plot_pareto_front_projections(pareto_3d, tmp_path / "proj", objective_names=["a", "b", "c"])
        assert [(ax.get_xlabel(), ax.get_ylabel()) for ax in axes] == [("a", "b"), ("a", "c"), ("b", "c")]
        np.testing.assert_array_equal(axes[0].collections[0].get_array(), pareto_3d[:, 2])
        np.testing.assert_array_equal(axes[2].collections[0].get_array(), pareto_3d[:, 0])
        assert (tmp_path / "proj.png").is_file()


class TestParallelCoordinates:
    def test_real_unit_axes_and_knee_line(self, tmp_path, pareto_2d, decision_vars):
        ax = plot_parallel_coordinates(
            pareto_2d,
            tmp_path / "parallel",
            decision_vars=decision_vars,
            lower_bounds=[0, 0],
            upper_bounds=[1, 1],
            parameter_names=["x", "y"],
            objective_names=["f1", "f2"],
        )
        assert [t.get_text() for t in ax.get_xticklabels()] == ["x", "y", "f1", "f2"]
        knee = calculate_knee_point(pareto_2d)
        expected = np.r_[decision_vars[knee], (pareto_2d[knee] - pareto_2d.min(0)) / np.ptp(pareto_2d, 0)]
        np.testing.assert_allclose(ax.lines[-1].get_ydata(), expected)
        assert {"0", "1", "5"} <= {t.get_text() for t in ax.texts}  # axis ends in real units
        assert (tmp_path / "parallel.pdf").is_file()


class TestSolutionTable:
    def test_rows_merge_designs_best_for_several_objectives(self):
        front = np.array([[0.0, 1.0, 1.0], [0.5, 0.5, 0.5], [1.0, 0.0, 0.0]])
        assert select_solutions(front, ["f1", "f2", "f3"]) == {
            "Knee point": 1,
            "Best f1": 0,
            "Best f2, f3": 2,
        }

    def test_table_shows_real_values(self, tmp_path, pareto_2d, decision_vars):
        fig = plot_solution_table(
            pareto_2d,
            decision_vars,
            tmp_path / "table",
            objective_names=["f1", "f2"],
            parameter_names=["x", "y"],
            maximize=[False, True],
        )
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert texts[:5] == ["", "f1", "f2", "x", "y"]
        assert texts[5:10] == ["Knee point", "2", "-3", "0.3", "0.7"]
        assert (tmp_path / "table.png").is_file()


class TestDecisionAlongFront:
    def test_positions_within_bounds_ordered_by_first_objective(self):
        front = np.array([[3.0, 0.0], [1.0, 2.0], [2.0, 1.0]])
        x = np.array([[10.0, 5.0], [0.0, 5.0], [5.0, 5.0]])
        ax = plot_decision_along_front(
            front, x, lower_bounds=[0, 0], upper_bounds=[10, 10], parameter_names=["a", "b"]
        )
        try:
            np.testing.assert_allclose(ax.images[0].get_array(), [[0, 0.5, 1], [0.5, 0.5, 0.5]])
            assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b"]
        finally:
            plt.close(ax.figure)


class TestProgress:
    def test_hypervolume_curve_counts_only_eligible_designs(self):
        F = np.array([[1.0, 1.0], [0.5, 0.5], [0.1, 0.1]])
        evaluations, hv = hypervolume_curve(F, np.array([2.0, 2.0]), np.array([True, True, False]))
        np.testing.assert_array_equal(evaluations, [1, 2, 3])
        np.testing.assert_allclose(hv, [1.0, 2.25, 2.25])  # the infeasible third design adds nothing

    def test_hypervolume_plot_has_one_line_per_run(self, tmp_path):
        a = np.array([[1.0, 1.0], [0.5, 0.5]])
        ax = plot_hypervolume_vs_evaluations([a, a[::-1]], tmp_path / "hv", labels=["A", "B"])
        assert len(ax.lines) == 2
        assert (tmp_path / "hv.png").is_file()

    def test_constraint_check_is_a_moving_share(self):
        ax = plot_constraint_check(np.r_[np.zeros(20, bool), np.ones(20, bool)], window=20)
        try:
            y = ax.lines[0].get_ydata()
            assert (y[0], y[-1]) == (0, 1)
        finally:
            plt.close(ax.figure)

    def test_run_monitor_is_one_png(self, tmp_path, pareto_2d):
        fig = plot_run_monitor(pareto_2d, tmp_path / "monitor", eligible=np.ones(5, bool))
        assert len(fig.axes) == 2
        assert (tmp_path / "monitor.png").is_file()
        assert not (tmp_path / "monitor.pdf").exists()


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
        """Only the publication style is accepted."""
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
        apply_style("publication", variant="clean")

    def test_medicalphysics_saves_at_exact_column_width(self, temp_dir):
        """The saved PNG is exactly 80 mm wide, since the journal won't rescale."""
        front = np.column_stack([np.linspace(0, 1, 5), np.linspace(1, 0, 5)])
        save_path = Path(temp_dir) / "medphys.png"
        try:
            apply_style("publication", variant="medicalphysics")
            plot_pareto_front_2d(front, save_path, show_knee_point=True)
        finally:
            apply_style("publication", variant="clean")
        width_px = mpimg.imread(save_path).shape[1]
        assert width_px / 600 * 25.4 == pytest.approx(80, abs=0.1)

    def test_medicalphysics_widths_snap_to_journal_columns(self):
        """Figures are 80 mm or 180 mm wide, never wider, so print text stays >= 10 pt."""
        from TopasMOO.plotting.style import DOUBLE_COL_WIDTH, SINGLE_COL_WIDTH, scale_figsize

        try:
            apply_style("publication", variant="medicalphysics")
            single = scale_figsize(SINGLE_COL_WIDTH, 2.8)
            double = scale_figsize(DOUBLE_COL_WIDTH, 5.2)
            grid = scale_figsize(3 * SINGLE_COL_WIDTH, 2.8)
        finally:
            apply_style("publication", variant="clean")
        assert single[0] * 25.4 == pytest.approx(80)
        assert double[0] * 25.4 == pytest.approx(180)
        assert grid[0] * 25.4 == pytest.approx(180)
        assert double[1] / double[0] == pytest.approx(5.2 / DOUBLE_COL_WIDTH)  # aspect kept
        assert grid[1] == pytest.approx(2.8)  # wide grids keep their height
        assert scale_figsize(DOUBLE_COL_WIDTH, 5.2) == (DOUBLE_COL_WIDTH, 5.2)  # other styles

    def test_medicalphysics_text_and_panel_labels(self, tmp_path, pareto_3d, decision_vars):
        """No text below 10 pt as printed; multi-panel parts are labeled (a), (b), ..."""
        from matplotlib.text import Text

        try:
            apply_style("publication", variant="medicalphysics")
            axes = plot_pareto_front_projections(pareto_3d, tmp_path / "proj")
            table = plot_solution_table(pareto_3d, decision_vars, tmp_path / "table")
            parallel = plot_parallel_coordinates(pareto_3d, tmp_path / "par", decision_vars=decision_vars)
        finally:
            apply_style("publication", variant="clean")
        for fig in (axes[0].figure, table, parallel.figure):
            sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
            assert min(sizes) >= 10
        assert [ax.get_title()[:3] for ax in axes] == ["(a)", "(b)", "(c)"]
        assert all(ax.title.get_wrap() for ax in axes)  # saved at exact width: wrap, don't clip

    @pytest.mark.parametrize("show_knee_point", [False, True])
    def test_medicalphysics_markers_and_legend_follow_style(self, show_knee_point):
        """Scatter edges and legend frame come from the style, not hardcoded."""
        from TopasMOO.plotting.style import marker_edge_width

        front = np.column_stack([np.linspace(0, 1, 5), np.linspace(1, 0, 5)])
        try:
            apply_style("publication", variant="medicalphysics")
            assert marker_edge_width(0.25) == 0
            ax = plot_pareto_front_2d(front, true_front=front, show_knee_point=show_knee_point)
            assert not ax.get_legend().get_frame_on()
            assert all((c.get_linewidths() == 0).all() for c in ax.collections)
            plt.close(ax.figure)

            apply_style("publication", variant="clean")
            assert marker_edge_width(0.25) > 0
        finally:
            apply_style("publication", variant="clean")


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
