"""Region model for the QVCache baseline: PCA projection to ``pdim``
dimensions, each split into ``nbuckets`` equal-width buckets between the
projected sample's minimum and maximum (QVCache: "dividing each dimension of
the reduced space into equal-sized buckets"). Values outside the sample range
fall into the edge buckets. Written in the binary format read by
QVCacheEngine."""
from __future__ import annotations

import numpy as np


def fit_region_model(sample: np.ndarray, path: str, pdim: int = 16, nbuckets: int = 8,
                     seed: int = 0, max_rows: int = 200_000) -> None:
    rng = np.random.default_rng(seed)
    X = np.asarray(sample, dtype=np.float64)
    if X.shape[0] > max_rows:
        X = X[rng.choice(X.shape[0], max_rows, replace=False)]
    d = X.shape[1]
    assert 1 <= pdim <= d and nbuckets >= 2
    mean = X.mean(axis=0)
    Xc = X - mean
    # principal directions = right singular vectors
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    comps = vt[:pdim]
    Z = Xc @ comps.T
    lo, hi = Z.min(axis=0), Z.max(axis=0)
    frac = np.arange(1, nbuckets) / nbuckets
    bounds = lo[:, None] + (hi - lo)[:, None] * frac[None, :]  # pdim x (nbuckets-1)
    with open(path, "wb") as f:
        f.write(b"QVRM")
        np.array([d, pdim, nbuckets], dtype=np.int32).tofile(f)
        mean.astype(np.float32).tofile(f)
        comps.astype(np.float32).tofile(f)
        np.ascontiguousarray(bounds, dtype=np.float32).tofile(f)
