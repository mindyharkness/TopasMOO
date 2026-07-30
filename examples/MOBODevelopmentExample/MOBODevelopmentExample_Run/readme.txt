
MOBO (BoTorch) development example on ZDT1 in testing_mode.

Install the optional extra first:
  uv sync --extra mobo

Choose MOBO when the evaluation budget is roughly below 500 simulations and
the parameter count is comfortably below ~15. Prefer NSGA-II for larger
budgets or higher-dimensional search spaces.
