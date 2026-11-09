#!/usr/bin/env python3
"""Builds a DiskANN in-memory (Vamana) index with diskannpy, DiskANN's default
parameters (R=64, L=100, alpha=1.2), and checks that VamanaExecutor loads it.
x86 host (diskannpy wheels are x86_64 only); the files are portable."""
import argparse
import os
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401
import trace_harness as th
from trace_harness.io import read_bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help=".fbin")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="ann")
    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--L", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    import diskannpy

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    diskannpy.build_memory_index(
        data=a.corpus, distance_metric="l2", index_directory=str(out), complexity=a.L,
        graph_degree=a.R, num_threads=a.threads or os.cpu_count(), alpha=a.alpha,
        vector_dtype=np.float32, index_prefix=a.prefix)
    ex = th.VamanaExecutor(str(out / a.prefix), a.corpus)
    n, d = read_bin(a.corpus).shape
    assert ex.size == n and ex.dim == d
    print(f"graph {out / a.prefix}: n={n} d={d} start={ex.entry_point} max_degree={ex.max_degree}")


if __name__ == "__main__":
    main()
