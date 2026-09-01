#!/usr/bin/env python3
"""Phase 3 benchmarks: constrained BNH + DTLZ2 (5-obj) acquisition timing."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from common import (
    bnh,
    bnh_decision_constraints_torch,
    bnh_feasible,
    dtlz2,
    hypervolume,
    hypervolume_reference_point,
    run_mobo,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    # --- Constrained BNH ---
    lower = np.array([0.0, 0.0])
    upper = np.array([5.0, 3.0])
    if args.quick:
        n_init, q, n_batches = 10, 2, 3
        num_restarts, raw_samples = 2, 32
    else:
        n_init, q, n_batches = 40, 4, 40
        num_restarts, raw_samples = 10, 512

    opt = run_mobo(
        objective_fn=bnh,
        lower=lower,
        upper=upper,
        n_obj=2,
        n_init=n_init,
        batch_size=q,
        n_batches=n_batches,
        seed=0,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        decision_constraints=bnh_decision_constraints_torch(),
    )
    # Filter reported Pareto to feasible (and assert none infeasible)
    X = opt.ParetoDecisionVars
    Y = opt.ParetoObjectives
    feas = bnh_feasible(X)
    n_infeasible = int((~feas).sum())
    Y_feas = Y[feas] if feas.any() else Y[:0]

    ref = hypervolume_reference_point(opt.train_Y)
    hv = hypervolume(Y_feas, ref) if len(Y_feas) else 0.0

    constrained = {
        "n_infeasible_in_pareto": n_infeasible,
        "n_pareto": int(len(Y)),
        "n_pareto_feasible": int(len(Y_feas)),
        "hv_feasible_pareto": hv,
    }
    print("constrained BNH", constrained)

    # --- DTLZ2 timing: ParEGO vs NEHVI at 5 objectives ---
    d = 8
    lower_d = np.zeros(d)
    upper_d = np.ones(d)
    n_obj = 5
    if args.quick:
        n_init_d, q_d, n_batches_d = 12, 2, 2
        nr, rs = 2, 32
    else:
        n_init_d, q_d, n_batches_d = 40, 2, 5
        nr, rs = 5, 128

    def obj(X):
        return dtlz2(X, n_obj=n_obj)

    times = {}
    for acq in ("qlognehvi", "qlognparego"):
        t0 = time.perf_counter()
        o = run_mobo(
            objective_fn=obj,
            lower=lower_d,
            upper=upper_d,
            n_obj=n_obj,
            n_init=n_init_d,
            batch_size=q_d,
            n_batches=n_batches_d,
            seed=1,
            acquisition=acq,
            num_restarts=nr,
            raw_samples=rs,
        )
        times[acq] = {
            "seconds": time.perf_counter() - t0,
            "resolved": o._acquisition_resolved,
            "n_obs": len(o.train_X),
        }
        print(acq, times[acq])

    # auto selection check
    o_auto = run_mobo(
        objective_fn=obj,
        lower=lower_d,
        upper=upper_d,
        n_obj=n_obj,
        n_init=min(n_init_d, 12),
        batch_size=1,
        n_batches=0,
        seed=2,
        acquisition="auto",
        num_restarts=1,
        raw_samples=16,
    )
    auto_resolved = o_auto._acquisition_resolved

    ratio = None
    if times["qlognehvi"]["seconds"] > 0:
        ratio = times["qlognparego"]["seconds"] / times["qlognehvi"]["seconds"]

    summary = {
        "constrained_bnh": constrained,
        "dtlz2_timing": times,
        "parego_over_nehvi_wall_time_ratio": ratio,
        "auto_selection_5obj": auto_resolved,
        "quick": args.quick,
    }
    out = OUT / ("phase3_quick.json" if args.quick else "phase3.json")
    out.write_text(json.dumps(summary, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
