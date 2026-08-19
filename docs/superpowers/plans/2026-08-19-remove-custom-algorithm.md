# Remove `custom_algorithm` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `NSGAII_Optimizer` always use NSGA-II and remove the misleading `custom_algorithm` API.

**Architecture:** Keep the existing `NSGAII_Optimizer` class and base-class keyword forwarding, but remove the custom algorithm parameter, state, and branch. Remove every source and documentation reference so custom algorithms are represented only by separate optimizer classes.

**Tech Stack:** Python 3, pymoo, pytest, Ruff.

## Global Constraints

- Hard-remove the API; do not deprecate or silently ignore it.
- Do not add a permanent or CI test.
- Run one temporary test and delete it after verification.
- Preserve the user's unrelated uncommitted NSGA-III work.

---

### Task 1: Remove the custom algorithm path

**Files:**
- Modify: `TopasMOO/optimizers.py:93-98,1230-1290,1489-1496`
- Modify: `docsrc/index.md:140-148,204-210`
- Temporary test: `/tmp/test_custom_algorithm_removed.py`

**Interfaces:**
- Consumes: `TopasMOOBaseClass.__init__(**kwds)` and `pymoo.algorithms.moo.nsga2.NSGA2`.
- Produces: `NSGAII_Optimizer.__init__(pop_size=20, seed=None, verbose=False, eliminate_duplicates=True, **kwds)` with `self.algorithm: NSGA2`.

- [ ] **Step 1: Write and run the temporary failing test**

Create `/tmp/test_custom_algorithm_removed.py`:

```python
import inspect

from TopasMOO.optimizers import NSGAII_Optimizer


def test_custom_algorithm_is_not_an_explicit_parameter():
    assert "custom_algorithm" not in inspect.signature(
        NSGAII_Optimizer.__init__
    ).parameters
```

Run: `.venv/bin/pytest -q /tmp/test_custom_algorithm_removed.py`

Expected before implementation: FAIL because `custom_algorithm` remains in the signature.

- [ ] **Step 2: Remove the production path**

In `TopasMOO/optimizers.py`, remove `custom_algorithm` from the NSGA-II signature and docstrings, remove `self.custom_algorithm`, and replace the conditional with unconditional construction:

```python
self.algorithm = NSGA2(
    pop_size=pop_size,
    sampling=_StartPointSampling(self.StartingValues),
    crossover=SBX(eta=15, prob=0.9),
    mutation=PM(eta=20),
    eliminate_duplicates=eliminate_duplicates,
)
```

Also remove stale `custom_algorithm` references from the base-class and NSGA-III docstrings.

- [ ] **Step 3: Remove user-documentation references**

Delete the `custom_algorithm` parameter bullets and descriptions from `docsrc/index.md`. Keep the guidance that additional algorithms should be implemented as separate classes.

- [ ] **Step 4: Verify the temporary test passes, then delete it**

Run: `.venv/bin/pytest -q /tmp/test_custom_algorithm_removed.py`

Expected: PASS.

Delete `/tmp/test_custom_algorithm_removed.py`; it must not become a repository test.

- [ ] **Step 5: Verify completeness and regressions**

Run:

```bash
rg "custom_algorithm" TopasMOO docsrc tests examples README.md QUICKSTART.md
.venv/bin/pytest -q tests/test_optimizers.py
.venv/bin/ruff check TopasMOO tests
```

Expected: no `custom_algorithm` matches, targeted tests pass, and Ruff reports `All checks passed!`.

- [ ] **Step 6: Inspect the final diff**

Run: `git diff -- TopasMOO/optimizers.py docsrc/index.md`

Expected: only the requested API/documentation removal in existing code; the user's unrelated NSGA-III changes remain intact.
