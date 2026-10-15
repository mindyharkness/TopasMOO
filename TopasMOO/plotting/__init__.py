"""TopasMOO plotting subpackage — publication-quality multi-objective visualizations."""

from .comprehensive import (
    ALL_PLOT_KEYS,
    DEFAULT_FINAL_PLOTS,
    GenerateComprehensiveVisualizations,
    RunData,
)
from .convergence import plot_parameter_convergence
from .correlation import plot_parameter_objective_correlation
from .decision import plot_decision_along_front
from .gp_correlation import plot_gp_prediction_correlation
from .parallel import plot_parallel_coordinates
from .pareto import (
    display_objectives,
    plot_pareto_front,
    plot_pareto_front_2d,
    plot_pareto_front_projections,
)
from .progress import (
    hypervolume_curve,
    plot_constraint_check,
    plot_hypervolume_vs_evaluations,
    plot_run_monitor,
)
from .style import (
    apply_style,
    available_publication_variants,
    publication_style,
    save_publication_figure,
)
from .table import plot_solution_table

__all__ = [
    # Style
    "apply_style",
    "available_publication_variants",
    "publication_style",
    "save_publication_figure",
    # Default: Pareto front
    "plot_pareto_front",
    "plot_pareto_front_2d",
    "plot_pareto_front_projections",
    "display_objectives",
    # Opt-in diagnostics
    "plot_hypervolume_vs_evaluations",
    "hypervolume_curve",
    "plot_constraint_check",
    "plot_solution_table",
    "plot_parallel_coordinates",
    "plot_decision_along_front",
    "plot_parameter_convergence",
    "plot_parameter_objective_correlation",
    "plot_gp_prediction_correlation",
    # During the run
    "plot_run_monitor",
    # Orchestration
    "GenerateComprehensiveVisualizations",
    "RunData",
    "DEFAULT_FINAL_PLOTS",
    "ALL_PLOT_KEYS",
]
