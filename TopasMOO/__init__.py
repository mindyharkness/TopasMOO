"""
TopasMOO: Multi-Objective Optimization for TOPAS Monte Carlo Simulations

TopasMOO extends the TopasOpt framework to handle multiple competing objectives
using pymoo algorithms, particularly NSGA-II.

Each optimization must import:
``NSGAII_Optimizer``
    The optimizer you construct and run (``.RunOptimization()``). This is the
    main entry point for most users.
``TopasMOOBaseClass``
    Base class to subclass if you want to drive a different pymoo algorithm.

The full visualization toolbox (Pareto fronts, petal diagrams, convergence plots,
etc.) lives in the TopasMOO.plotting subpackage; only the style entry
points are re-exported here for convenience. Pareto-analysis metrics live in
TopasMOO.metrics and log helpers in TopasMOO.io.
"""

import importlib.metadata

__author__ = "Mindy Harkness"
try:
    __version__ = importlib.metadata.version("topasmoo")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.2.0-dev"

# Exceptions
from .exceptions import (
    InvalidParameterError,
    MalformedOutputError,
    ObjectiveFunctionError,
    TopasExecutionError,
    TopasMOOError,
)

# I/O utilities
from .io import LogParetoFrontToFile, ReadInMultiObjectiveLogFile

# Metrics
from .metrics import (
    calculate_crowding_distance,
    calculate_dominance_rank,
    calculate_knee_point,
)

# Core optimizer classes (TopasProblem is an internal pymoo adapter that stays
# in TopasMOO.optimizers).
from .optimizers import NSGAII_Optimizer, TopasMOOBaseClass

# Plotting style entry points (the full plotting API lives in TopasMOO.plotting).
from .plotting import (
    apply_style,
    available_publication_variants,
    publication_style,
    save_publication_figure,
)

__all__ = [
    # Core classes
    "NSGAII_Optimizer",
    "TopasMOOBaseClass",
    # Exceptions
    "TopasMOOError",
    "TopasExecutionError",
    "InvalidParameterError",
    "ObjectiveFunctionError",
    "MalformedOutputError",
    # Metrics
    "calculate_knee_point",
    "calculate_crowding_distance",
    "calculate_dominance_rank",
    # IO
    "ReadInMultiObjectiveLogFile",
    "LogParetoFrontToFile",
    # Plotting style entry points (full plotting API under TopasMOO.plotting)
    "apply_style",
    "available_publication_variants",
    "publication_style",
    "save_publication_figure",
]
