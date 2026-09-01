# MOBO Development Example

Short Bayesian multi-objective campaign (BoTorch `qLogNEHVI` / `qLogNParEGO`) on the analytic ZDT1 problem in `testing_mode`.

## Setup

```bash
uv sync --extra mobo --extra dev
uv run python examples/MOBODevelopmentExample/MOBODevelopmentExample_main.py
```

## When to use MOBO vs NSGA-II

- Prefer **MOBO** when each evaluation is expensive and the budget is roughly **below 500** simulations, with parameter count **below ~15**.
- Prefer **NSGA-II** for larger budgets, higher-dimensional spaces, or when you do not want the optional BoTorch stack.

Production TOPAS studies should increase `n_init` and `n_generations` (acquisition batches) beyond the values in this example.

## GP prediction correlation

The completed run writes `logs/FinalResults/GPPredictionCorrelation.pdf` and
`.png`. Each objective gets an observed-versus-predicted panel using the GP
posterior mean saved when the candidate was proposed. The Pearson and Spearman
values therefore assess prospective surrogate quality; the Sobol initial
design is omitted because it has no pre-evaluation GP prediction.
