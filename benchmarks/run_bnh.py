#!/usr/bin/env python3
"""BNH unconstrained MOBO benchmark vs NSGA-II and Sobol.

Acceptance (full): MOBO HV within 5% of NSGA-II at 5000 evaluations, using 300
MOBO evaluations, over 5 seeds.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from common import (
    bnh,
    hypervolume,
    hypervolume_reference_point,
    nd_front,
    run_mobo,
    run_nsga2_pymoo,
    sobol_sample,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seeds", type=int, default=None)
    args = p.parse_args()

    lower = np.array([0.0, 0.0])
    upper = np.array([5.0, 3.0])

    if args.quick:
        n_init, q, n_batches = 12, 2, 4
        seeds = list(range(args.seeds or 2))
        num_restarts, raw_samples = 3, 64
        nsga_pop, nsga_gen = 20, 10  # 200 evals stand-in
    else:
        n_init, q, n_batches = 40, 4, 65  # 40 + 260 = 300
        seeds = list(range(args.seeds or 5))
        num_restarts, raw_samples = 10, 512
        nsga_pop, nsga_gen = 50, 100  # 5000 evals

    budget = n_init + q * n_batches
    rows = []
    ref = None
    for seed in seeds:
        t0 = time.perf_counter()
        opt = run_mobo(
            objective_fn=bnh,
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
        mobo_Y = opt.train_Y
        mobo_s = time.perf_counter() - t0

        Xs = sobol_sample(budget, lower, upper, seed=seed + 2000)
        sobol_Y = bnh(Xs)

        _, nsga_F = run_nsga2_pymoo(
            bnh, lower, upper, 2, nsga_pop, nsga_gen, seed
        )

        all_Y = np.vstack([mobo_Y, sobol_Y, nsga_F])
        if ref is None:
            # Same rule the optimizers use for their own hypervolume histories,
            # so these numbers stay comparable with what a run reports.
            ref = hypervolume_reference_point(all_Y)

        row = {
            "seed": seed,
            "mobo_hv": hypervolume(mobo_Y, ref),
            "sobol_hv": hypervolume(sobol_Y, ref),
            "nsga_hv": hypervolume(nsga_F, ref),
            "mobo_seconds": mobo_s,
            "n_pareto_mobo": int(len(nd_front(mobo_Y))),
        }
        rows.append(row)
        print(row)

    med_mobo = float(np.median([r["mobo_hv"] for r in rows]))
    med_nsga = float(np.median([r["nsga_hv"] for r in rows]))
    summary = {
        "config": {
            "n_init": n_init,
            "batch_size": q,
            "n_batches": n_batches,
            "budget": budget,
            "nsga_evals": nsga_pop * nsga_gen,
            "quick": args.quick,
        },
        "median_mobo_hv": med_mobo,
        "median_sobol_hv": float(np.median([r["sobol_hv"] for r in rows])),
        "median_nsga_hv": med_nsga,
        "mobo_vs_nsga_ratio": med_mobo / med_nsga if med_nsga else None,
        "rows": rows,
    }
    out = OUT / ("bnh_quick.json" if args.quick else "bnh.json")
    out.write_text(json.dumps(summary, indent=2))
    print("Wrote", out)
    print(summary)


if __name__ == "__main__":
    main()
