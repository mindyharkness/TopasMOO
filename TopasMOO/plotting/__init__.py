"""TopasMOO plotting subpackage — publication-quality multi-objective visualizations."""

from .comprehensive import (
    ALL_PLOT_KEYS,
    DEFAULT_FINAL_PLOTS,
    GenerateComprehensiveVisualizations,
    RunData,
)
from .convergence import plot_objective_convergence, plot_parameter_convergence
from .correlation import plot_parameter_objective_correlation
from .decision import plot_decision_heatmap
from .hypervolume import plot_hypervolume_convergence
from .parallel import plot_parallel_coordinates
from .pareto import (
    plot_pareto_front,
    plot_pareto_front_2d,
    plot_pareto_front_3d,
    plot_pareto_front_projections,
)
from .petal import plot_petal_diagram_multi, plot_petal_diagram_single
from .population import plot_population_evolution
from .style import (
    apply_style,
    available_publication_variants,
    publication_style,
    save_publication_figure,
)

__all__ = [
    # Style
    "apply_style",
    "available_publication_variants",
    "publication_style",
    "save_publication_figure",
    # Pareto front
    "plot_pareto_front",
    "plot_pareto_front_2d",
    "plot_pareto_front_3d",
    "plot_pareto_front_projections",
    # Convergence
    "plot_objective_convergence",
    "plot_parameter_convergence",
    "plot_hypervolume_convergence",
    # Solution analysis
    "plot_petal_diagram_single",
    "plot_petal_diagram_multi",
    "plot_parameter_objective_correlation",
    "plot_decision_heatmap",
    "plot_parallel_coordinates",
    # Population dynamics
    "plot_population_evolution",
    # Orchestration
    "GenerateComprehensiveVisualizations",
    "RunData",
    "DEFAULT_FINAL_PLOTS",
    "ALL_PLOT_KEYS",
]
