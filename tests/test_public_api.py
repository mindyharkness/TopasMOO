from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import TopasMOO
from TopasMOO.exceptions import MalformedOutputError
from TopasMOO.io import ReadInMultiObjectiveLogFile
from TopasMOO.plotting import (
    DEFAULT_FINAL_PLOTS,
    plot_gp_prediction_correlation,
    plot_parallel_coordinates,
)
from TopasMOO.plotting.comprehensive import (
    GenerateComprehensiveVisualizations,
    RunData,
    _resolve_final_plots,
)
from TopasMOO.plotting.style import (
    apply_style,
    available_publication_variants,
    publication_style,
)


def test_default_final_plots_are_lean() -> None:
    assert DEFAULT_FINAL_PLOTS == {
        "pareto",
        "convergence",
        "parameter_convergence",
        "hypervolume",
    }


def test_resolve_final_plots_treats_bare_key_as_singleton() -> None:
    plots, explicit = _resolve_final_plots("pareto")
    assert plots == {"pareto"}
    assert explicit
    # Character-set footgun must not win for a recognized key.
    assert plots != set("pareto")


def test_resolve_final_plots_aliases_and_iterable() -> None:
    assert _resolve_final_plots(None)[0] == set(DEFAULT_FINAL_PLOTS)
    assert _resolve_final_plots("default")[0] == set(DEFAULT_FINAL_PLOTS)
    assert "parallel" in _resolve_final_plots("all")[0]
    assert _resolve_final_plots(["pareto", "parallel"])[0] == {"pareto", "parallel"}
    assert _resolve_final_plots("not_a_key")[0] == set()
    assert _resolve_final_plots("gp_correlation")[0] == {"gp_correlation"}


def test_generate_comprehensive_accepts_single_key_string(tmp_path) -> None:
    run = RunData(
        pareto_objectives=np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]]),
        n_objectives=2,
        parameter_names=["x1", "x2"],
    )
    GenerateComprehensiveVisualizations(run, tmp_path, final_plots="pareto")
    stems = {path.stem for path in tmp_path.iterdir()}
    assert "ParetoFront_Final" in stems
    assert "ParallelCoordinates_Final" not in stems


def test_plot_pareto_front_dispatches_by_objective_count(tmp_path) -> None:
    from TopasMOO.plotting import plot_pareto_front

    front2 = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
    plot_pareto_front(front2, tmp_path / "p2", show_knee_point=True)
    assert (tmp_path / "p2.png").is_file()

    front3 = np.array([[0.1, 0.2, 0.9], [0.5, 0.5, 0.5], [0.9, 0.2, 0.1]])
    plot_pareto_front(front3, tmp_path / "p3", show_knee_point=True)
    assert (tmp_path / "p3.png").is_file()


def test_plotting_public_api_exports_parallel_coordinates() -> None:
    pareto = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
    ax = plot_parallel_coordinates(pareto)
    assert ax is not None
    plt.close(ax.figure)


def test_plotting_public_api_exports_gp_correlation() -> None:
    observed = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    axes = plot_gp_prediction_correlation(observed, observed + 0.1)
    assert axes.shape == (1, 2)
    plt.close(axes[0, 0].figure)


def test_generate_comprehensive_adds_gp_correlation_for_mobo_default(tmp_path) -> None:
    observed = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    run = RunData(
        pareto_objectives=observed,
        n_objectives=2,
        parameter_names=["x1", "x2"],
        observed_objectives=observed,
        gp_prediction_history=observed + 0.1,
        failed_mask=np.zeros(3, dtype=bool),
    )

    GenerateComprehensiveVisualizations(run, tmp_path)

    assert (tmp_path / "GPPredictionCorrelation.pdf").is_file()
    assert (tmp_path / "GPPredictionCorrelation.png").is_file()


def test_root_package_exposes_publication_variants() -> None:
    assert "available_publication_variants" in TopasMOO.__all__
    assert set(TopasMOO.available_publication_variants()) == {
        "clean",
        "nature",
        "ieee",
        "medicalphysics",
    }


def test_root_package_exports_nsga3_optimizer() -> None:
    from TopasMOO.optimizers import NSGAIII_Optimizer

    assert "NSGAIII_Optimizer" in TopasMOO.__all__
    assert TopasMOO.NSGAIII_Optimizer is NSGAIII_Optimizer


def test_utilities_module_importable() -> None:
    import TopasMOO.utilities as utilities

    assert callable(utilities._import_from_absolute_path)
    assert callable(utilities._load_user_callable)


def test_read_log_raises_file_not_found_error_for_missing_file(tmp_path) -> None:
    missing = tmp_path / "does_not_exist.txt"
    with pytest.raises(FileNotFoundError):
        ReadInMultiObjectiveLogFile(str(missing))


def test_read_log_raises_malformed_output_on_bad_value(tmp_path) -> None:
    bad_log = tmp_path / "bad.txt"
    bad_log.write_text(
        "Iteration: 0, x1: 0.5, x2: 0.5, ObjectiveFunction_1: not_a_number\n"
    )
    with pytest.raises(MalformedOutputError):
        ReadInMultiObjectiveLogFile(str(bad_log))


def test_publication_variants_listed() -> None:
    variants = available_publication_variants()
    assert set(variants) == {"clean", "nature", "ieee", "medicalphysics"}


@pytest.mark.parametrize("variant", ["clean", "nature", "ieee", "medicalphysics"])
def test_apply_publication_variants(variant: str) -> None:
    apply_style("publication", variant=variant)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    plt.close(fig)
    apply_style("fast")


def test_publication_style_context_manager_with_variant() -> None:
    with publication_style("publication", variant="ieee"):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        plt.close(fig)


def test_removed_poster_variant_raises() -> None:
    with pytest.raises(ValueError, match="Unknown publication variant 'poster'"):
        apply_style("publication", variant="poster")
