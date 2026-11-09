#!/usr/bin/env python3
"""Exact top-K for a workload against a corpus (exhaustive float32 scan with
float64 rerank; run outside the memory cap). Writes gt_ids.npy, gt_dist.npy,
and reports the locality-ratio distribution."""
import argparse
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401
from trace_harness.gt import exact_knn
from trace_harness.io import read_bin, save_json
from trace_harness.streams import locality_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    X = read_bin(a.corpus)
    w = Path(a.workload)
    Q = np.load(w / "queries.npy")
    S = np.load(w / "sessions.npy")
    ids, dist = exact_knn(X, Q, K=a.K)
    np.save(w / "gt_ids.npy", ids)
    np.save(w / "gt_dist.npy", dist)
    rho = locality_ratio(Q, S, dist[:, a.k - 1])
    stats = {"corpus": a.corpus, "K": a.K, "median_rho": float(np.median(rho)),
             "frac_rho_below_0.5": float((rho < 0.5).mean())}
    save_json(w / "gt_meta.json", stats)
    print(stats)


if __name__ == "__main__":
    main()
