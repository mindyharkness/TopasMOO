# Remove `custom_algorithm` from NSGA-II

## Design

`NSGAII_Optimizer` will represent only the built-in NSGA-II implementation. Its
constructor will no longer accept or store `custom_algorithm`; initialization
will always construct the configured `pymoo.algorithms.moo.nsga2.NSGA2`
instance.

All references advertising `custom_algorithm` will be removed from source
docstrings and user documentation. Custom optimization algorithms remain
supported through separate optimizer classes derived from `TopasMOOBaseClass`.

## Compatibility

This is an intentional breaking API change. Existing callers that pass
`custom_algorithm` will receive `TypeError` rather than silently selecting a
different algorithm.

## Verification

No permanent regression test will be added. A temporary test will verify that:

1. the constructor signature no longer includes `custom_algorithm`;
2. passing `custom_algorithm` is rejected; and
3. normal construction still creates an `NSGA2` instance.

The temporary test will be deleted after it passes. Existing targeted tests,
lint, and diagnostics will then be run.
