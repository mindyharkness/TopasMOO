# Multi-Objective Aperture Optimization Example

This example demonstrates multi-objective optimization of a collimator design using TopasMOO. It is adapted from TopasOpt. 

## Objectives

This optimization simultaneously optimizes for two competing objectives:

1. **Dose Profile Accuracy**: Minimize the error between the actual dose distribution and a target (ground truth) distribution
2. **Beam Efficiency**: Maximize the number of particles transmitted through the collimator (minimize dose loss)

## Parameters Being Optimized

- `UpStreamApertureRadius`: Radius of the upstream aperture opening (1-3 mm)
- `DownStreamApertureRadius`: Radius of the downstream aperture opening (1-3 mm)
- `CollimatorThickness`: Thickness of the collimator (10-40 mm)

## Running the Example

```bash
python ApertureOptimization_main.py
```

**Note**: You need a working TOPAS installation. Update the `TopasLocation` parameter in `ApertureOptimization_main.py` to point to your TOPAS installation.

## Output

The optimization will create a directory structure with:

- **logs/**: Optimization logs, convergence plots, Pareto front visualizations
- **Results/**: TOPAS simulation outputs
- **TopasScripts/**: Generated TOPAS input files

## References

- Original single-objective version: See TopasOpt examples
- NSGA-II algorithm: Deb et al. (2002), "A Fast and Elitist Multiobjective Genetic Algorithm"
