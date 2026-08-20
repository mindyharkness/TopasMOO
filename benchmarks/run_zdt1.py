#!/usr/bin/env python3
"""ZDT1 MOBO benchmark (n_var=8, not the default 30).

A 30-variable GP with a few hundred points previously failed in this codebase;
``n_var`` is reduced to 8 deliberately.

Acceptance (full mode): with ``n_init=80``, ``q=4``, 30 batches (200 evaluations),
median IGD over 5 seeds must be below the median IGD of NSGA-II at the same
total evaluation count. Includes a Sobol-only control.

Usage::

    uv run python benchmarks/run_zdt1.py            # full (slow)
    uv run python benchmarks/run_zdt1.py --quick    # CI-friendly smoke numbers
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from common import (
    igd,
    run_mobo,
    run_nsga2_pymoo,
    sobol_sample,
    zdt1,
    zdt1_true_front,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seeds", type=int, default=None)
    args = p.parse_args()

    d = 8
    lower = np.zeros(d)
    upper = np.ones(d)
    true_front = zdt1_true_front(200)

    if args.quick:
        n_init, q, n_batches = 16, 2, 4  # 16 + 8 = 24 evals
        seeds = list(range(args.seeds or 2))
        num_restarts, raw_samples = 3, 64
        pop_size, n_gen = 8, 3
    else:
        n_init, q, n_batches = 80, 4, 30  # 80 + 120 = 200
        seeds = list(range(args.seeds or 5))
        # Slightly reduced restarts/raw_samples vs library defaults for numerical
        # stability on ZDT1 with ModelListGP; production MOBOOptimizer defaults
        # remain num_restarts=10, raw_samples=512.
        num_restarts, raw_samples = 5, 128
        # NSGA-II at same total evals: pop * n_gen ≈ 200
        pop_size, n_gen = 20, 10

    budget = n_init + q * n_batches
    rows = []
    for seed in seeds:
        t0 = time.perf_counter()
        opt = run_mobo(
            objective_fn=zdt1,
            lower=lower,
            upper=upper,
            n_obj=2,
            n_init=n_init,
            batch_size=q,
            n_batches=n_batches,
            seed=seed,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        mobo_igd = igd(opt.train_Y, true_front)
        mobo_s = time.perf_counter() - t0

        Xs = sobol_sample(budget, lower, upper, seed=seed + 1000)
        Ys = zdt1(Xs)
        sobol_igd = igd(Ys, true_front)

        Xn, Fn = run_nsga2_pymoo(zdt1, lower, upper, 2, pop_size, n_gen, seed)
        # Use all evaluated individuals if history unavailable: approximate with final front
        nsga_igd = igd(Fn, true_front)

        row = {
            "seed": seed,
            "mobo_igd": mobo_igd,
            "sobol_igd": sobol_igd,
            "nsga_igd": nsga_igd,
            "mobo_seconds": mobo_s,
            "budget": budget,
        }
        rows.append(row)
        print(row)

    summary = {
        "config": {
            "n_var": d,
            "n_init": n_init,
            "batch_size": q,
            "n_batches": n_batches,
            "budget": budget,
            "quick": args.quick,
        },
        "median_mobo_igd": float(np.median([r["mobo_igd"] for r in rows])),
        "median_sobol_igd": float(np.median([r["sobol_igd"] for r in rows])),
        "median_nsga_igd": float(np.median([r["nsga_igd"] for r in rows])),
        "rows": rows,
    }
    out = OUT / ("zdt1_quick.json" if args.quick else "zdt1.json")
    out.write_text(json.dumps(summary, indent=2))
    print("Wrote", out)
    print(
        "medians: MOBO={:.6f} NSGA={:.6f} Sobol={:.6f}".format(
            summary["median_mobo_igd"],
            summary["median_nsga_igd"],
            summary["median_sobol_igd"],
        )
    )


if __name__ == "__main__":
    main()
