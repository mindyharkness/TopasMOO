# ZDT1 Benchmark Example

This example validates TopasMOO against ZDT1, a standard multi-objective
benchmark with a known analytical Pareto front.

## Purpose

- Verify optimizer performance against a known solution.
- Confirm a TopasMOO installation works correctly.
- Demonstrate the multi-objective optimization workflow.
- Provide a starting point for custom objective functions.

## ZDT1

The example uses five decision variables in `[0, 1]` and minimizes two
objectives:

- `f1(x) = x1`
- `f2(x) = g(x) * (1 - sqrt(f1 / g))`
- `g(x) = 1 + 9 * sum(x2...x5) / 4`

The analytical Pareto front is `f2 = 1 - sqrt(f1)` for `f1` in `[0, 1]`.

## Files

- `DevelopmentExample_main.py` — runs the optimization and validation
- `TopasObjectiveFunction.py` — implements the ZDT1 objectives
- `GenerateTopasScripts.py` — stores parameters for testing mode
- `../ValidationMetrics.py` — shared concise numerical validation

## Running the example

From the repository root:

```bash
uv run python examples/DevelopmentExample/DevelopmentExample_main.py
```

The example does not require TOPAS because it uses TopasMOO's testing mode.

## Validation output

After writing the standard TopasMOO results, the example makes one validation
call:

```python
summary = generate_zdt1_validation(results, output_dir)
```

This creates three files in `DevelopmentExample_MOO/validation/`:

- `zdt1_validation.png` — polished visual summary
- `zdt1_validation.pdf` — publication-ready vector version
- `zdt1_validation.txt` — concise numerical report

The figure compares the obtained solutions with the analytical ZDT1 front and
shows:

- number of Pareto solutions
- inverted generational distance (IGD)
- hypervolume
- maximum error from the analytical front
- overall validation status

The returned `ValidationSummary` provides the same values programmatically.

## Validation criteria

Validation passes when:

- IGD is at most `0.05`
- maximum front error is at most `0.05`

IGD measures how well the obtained solutions cover the analytical front.
Maximum front error measures convergence to `f2 = 1 - sqrt(f1)`. Lower values
are better for both.

Hypervolume measures dominated objective space and is higher when convergence
and coverage improve. It uses the explicit fixed reference point `[1.1, 1.1]`
and is supporting information rather than a pass criterion.

General-purpose plots and optimization logs remain available in
`DevelopmentExample_MOO/logs/`.

## References

- Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of multiobjective
  evolutionary algorithms: Empirical results. *Evolutionary Computation*,
  8(2), 173–195.
- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist
  multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on
  Evolutionary Computation*, 6(2), 182–197.
- Van Veldhuizen, D. A. (1999). *Multiobjective evolutionary algorithms:
  Classifications, analyses, and new innovations*. PhD thesis.
