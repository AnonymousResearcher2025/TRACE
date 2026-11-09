#!/usr/bin/env python3
"""Fits the QVCache region model (PCA-16, 8 buckets per dimension)."""
import argparse

import numpy as np

from _common import ROOT  # noqa: F401
from trace_harness.io import read_bin
from trace_harness.qvcache_model import fit_region_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pdim", type=int, default=16)
    ap.add_argument("--nbuckets", type=int, default=8)
    ap.add_argument("--sample", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    X = read_bin(a.corpus)
    rng = np.random.default_rng(a.seed)
    rows = np.sort(rng.choice(X.shape[0], min(a.sample, X.shape[0]), replace=False))
    fit_region_model(np.asarray(X[rows]), a.out, a.pdim, a.nbuckets, a.seed)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
