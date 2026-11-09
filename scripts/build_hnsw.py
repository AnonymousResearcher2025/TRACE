#!/usr/bin/env python3
"""Builds an hnswlib index (labels = corpus row ids)."""
import argparse
import time

import numpy as np

from _common import ROOT  # noqa: F401
from trace_harness.io import read_bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help=".fbin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=100)
    a = ap.parse_args()
    import hnswlib
    import os

    X = read_bin(a.corpus)
    n, d = X.shape
    norms = np.linalg.norm(np.asarray(X[: min(n, 100000)], dtype=np.float64), axis=1)
    assert np.abs(norms - 1).max() < 1e-3, "corpus must be unit-normalized"
    idx = hnswlib.Index(space="l2", dim=d)
    idx.init_index(max_elements=n, ef_construction=a.efc, M=a.M, random_seed=a.seed)
    t = time.time()
    threads = a.threads or os.cpu_count()
    for s in range(0, n, 1_000_000):
        idx.add_items(np.asarray(X[s:s + 1_000_000]), np.arange(s, min(n, s + 1_000_000)),
                      num_threads=threads)
        print(f"{min(n, s + 1_000_000):,}/{n:,} ({time.time() - t:.0f}s)", flush=True)
    idx.save_index(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
