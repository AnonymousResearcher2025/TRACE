#!/usr/bin/env python3
"""Trains the DARTH-style recall predictor for one tier on its tuning split
(disjoint from the replay) and writes the model file named in the config.

By default the training list size is twice the cold-search L tuned to the
target recall (cached in results/tuning/<tier>/cold.json, tuned if absent),
the same L at which the baseline runs."""
import argparse
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401
from trace_harness import darth
from trace_harness.io import Workload
from trace_harness.runner import Tier, context, executor, load_config, split, tuned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--L", type=int, default=None,
                    help="search list size for training (default: 2x the cold tuned L)")
    ap.add_argument("--results", default="results", help="results directory (tuning cache)")
    ap.add_argument("--interval", type=int, default=64)
    a = ap.parse_args()
    cfg = load_config(a.config)
    tier = Tier.from_dict(a.tier, cfg["tiers"][a.tier])
    assert tier.darth_model, "set darth_model in the tier config"
    wl = Workload(tier.workload)
    sp = split(wl, int(cfg.get("tune_queries", 5000)), int(cfg.get("warmup", 2000)))
    ex = executor(tier)
    L = a.L
    if L is None:
        ctx = context(tier, ex=ex, k=int(cfg.get("k", 10)))
        target = float(cfg.get("target_recall", 0.95))
        L = 2 * tuned(Path(a.results), tier, ctx, "cold", sp, target)["L"]
        print(f"training at L={L} (2x the cold tuned L)")
    X, y = darth.collect(ex, np.ascontiguousarray(sp.tune.Q), sp.tune.sessions,
                         sp.tune.gt_ids, k=int(cfg.get("k", 10)), L=L,
                         interval=a.interval, beam_width=tier.beam_width)
    print(f"{len(y)} checkpoints, mean label {y.mean():.3f}")
    Path(tier.darth_model).parent.mkdir(parents=True, exist_ok=True)
    darth.train(X, y, tier.darth_model)
    darth.save_meta(tier.darth_model + ".json", L=L, interval=a.interval, n=len(y))
    print("wrote", tier.darth_model)


if __name__ == "__main__":
    main()
