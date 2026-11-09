"""Workloads and reference values for the fidelity checks of the two
baselines reimplemented on TRACE's executors, distance kernel, and counters.

QVCache (Göçer et al., arXiv:2602.02057, Sec. 5): queries are split into
N_split disjoint subsets; every query gets a perturbed copy
q' = (1 - eta) q + eta r, with r a random data vector. A window of
WINDOW_SIZE perturbed splits is shuffled and dispatched N_repeat times, then
``stride`` splits are replaced; a round repeats the cycle with fresh copies.
With 10K queries and the paper's settings (10, 0.01, 3, 4, 1, 1) the stream
has 84,000 queries, the total the paper reports.

CatapultDB (arXiv:2603.02164, Sec. 4.3): 150,000 queries drawn uniformly
from [-1, 1]^d against MedCPT embeddings of the first million PubMed
documents, the no-locality case. The paper's other workloads need its 35M
document corpus or LLM-paraphrased questions.
"""
from __future__ import annotations

import numpy as np

QVCACHE_DEFAULTS = dict(n_split=10, eta=0.01, n_repeat=3, window=4, stride=1, n_round=1)

# Values reported in the papers, kept next to our measurements in the output.
QVCACHE_REPORTED = {
    "source": "QVCache arXiv:2602.02057, Sec. 6.2 / Fig. 5 (DiskANN backend, k=10)",
    "recall_drop_pct": [2.0, 5.0],
    "hit_ratio_drop_at_shift_pct": 25.0,
    "hit_latency_ms": [0.1, 1.0],
    "p50_reduction_x": [60.0, 300.0],
}
CATAPULT_REPORTED = {
    "source": "CatapultDB arXiv:2603.02164, Sec. 4.3 (Uniform, t=4)",
    1: {"hops_reduction_pct": 35.7, "usage_pct": 98.1},
    16: {"hops_reduction_pct": 8.1, "usage_pct": 97.6},
}


def qvcache_stream(Q: np.ndarray, X: np.ndarray, n_split: int = 10, eta: float = 0.01,
                   n_repeat: int = 3, window: int = 4, stride: int = 1, n_round: int = 1,
                   seed: int = 0):
    """Returns (queries, dispatch index per query, base query index per query).
    Splits are contiguous ranges of the query file."""
    assert 1 <= window <= n_split and stride >= 1 and n_repeat >= 1 and n_round >= 1
    rng = np.random.default_rng(seed)
    splits = np.array_split(np.arange(len(Q)), n_split)
    Qd = np.asarray(Q, dtype=np.float64)
    out_q, out_disp, out_base = [], [], []
    disp = 0
    for _ in range(n_round):
        copies = []  # one perturbed copy per split and round
        for idx in splits:
            r = np.asarray(X[np.sort(rng.integers(0, X.shape[0], size=len(idx)))],
                           dtype=np.float64)
            copies.append(((1.0 - eta) * Qd[idx] + eta * r).astype(np.float32))
        for start in range(0, n_split - window + 1, stride):
            members = list(range(start, start + window))
            W = np.concatenate([copies[s] for s in members])
            B = np.concatenate([splits[s] for s in members])
            for _ in range(n_repeat):
                p = rng.permutation(len(W))
                out_q.append(W[p])
                out_base.append(B[p])
                out_disp.append(np.full(len(W), disp, dtype=np.int64))
                disp += 1
    return (np.ascontiguousarray(np.concatenate(out_q)), np.concatenate(out_disp),
            np.concatenate(out_base))


def uniform_stream(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n, d)).astype(np.float32)


def per_group(values: np.ndarray, groups: np.ndarray) -> list[float]:
    """Mean of values per group id 0..G-1."""
    G = int(groups.max()) + 1
    s = np.bincount(groups, weights=values.astype(np.float64), minlength=G)
    c = np.bincount(groups, minlength=G)
    return (s / np.maximum(c, 1)).tolist()


def shift_drop_pct(hit_by_dispatch: list[float], n_repeat: int) -> float:
    """Mean relative drop of the hit ratio at a window change: the first
    dispatch after a slide versus the last dispatch before it."""
    h = np.asarray(hit_by_dispatch)
    drops = [100.0 * (h[j - 1] - h[j]) / h[j - 1]
             for j in range(n_repeat, len(h), n_repeat) if h[j - 1] > 0]
    return float(np.mean(drops)) if drops else float("nan")
