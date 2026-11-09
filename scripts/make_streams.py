#!/usr/bin/env python3
"""Derives the locality-sweep and stream-dynamics workloads from a base video
workload and its corpus, and computes their ground truth.

Writes <out>/locality/<bin>/, <out>/locality/shuffled/, and
<out>/dynamics/{natural,alternating,abrupt,oscillating,interleaved}/, and
<out>/extra_workloads.json. configs/aws.yaml lists these paths for
--out ${data_root}/workloads/derived."""
import argparse
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401
from trace_harness import streams
from trace_harness.gt import exact_knn
from trace_harness.io import Workload, read_bin, save_json, save_workload

BINS = [(0.0, 0.125), (0.125, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0)]


def write(root, Q, S, X, meta, K):
    save_workload(root, Q, S, meta)
    ids, dist = exact_knn(X, Q, K=K)
    np.save(Path(root) / "gt_ids.npy", ids)
    np.save(Path(root) / "gt_dist.npy", dist)
    print("wrote", root, len(Q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--base", required=True, help="base video workload (with GT)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-bin", type=int, default=5000)
    ap.add_argument("--dyn-queries", type=int, default=20000)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    X = read_bin(a.corpus)
    base = Workload(a.base)
    Q, S = np.asarray(base.Q), base.sessions
    out = Path(a.out)
    cfg = {"locality": {}, "dynamics": {}}
    for lo, hi in BINS:
        name = f"rho_{lo:g}_{hi:g}"
        q, s = streams.controlled_locality(X, Q, lo, hi, a.per_bin, seed=a.seed)
        write(out / "locality" / name, q, s, X, {"rho": [lo, hi]}, a.K)
        cfg["locality"][name] = str(out / "locality" / name)
    q, s = streams.shuffled(Q[: a.per_bin], seed=a.seed)
    write(out / "locality" / "shuffled", q, s, X, {"shuffled": True}, a.K)
    cfg["locality"]["shuffled"] = str(out / "locality" / "shuffled")

    n = a.dyn_queries
    dyn = {
        "natural": (Q[:n], S[:n]),
        "alternating": streams.alternating(Q, S, block=16, seed=a.seed),
        "abrupt": streams.abrupt(Q[:n], S[:n], clips_per_session=10),
        "oscillating": streams.oscillating(X, Q, a.tau, n, seed=a.seed),
        "interleaved": streams.interleaved(Q[:n], S[:n], n=8),
    }
    for name, (q, s) in dyn.items():
        q, s = q[:n], s[:n]
        write(out / "dynamics" / name, np.ascontiguousarray(q), s, X, {"stream": name}, a.K)
        cfg["dynamics"][name] = str(out / "dynamics" / name)
    save_json(out / "extra_workloads.json", cfg)
    print("extra_workloads:", cfg)


if __name__ == "__main__":
    main()
