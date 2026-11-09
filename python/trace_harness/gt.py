"""Exact ground truth (exhaustive float32 scan, float64 rerank)."""
from __future__ import annotations

import numpy as np


def exact_knn(X: np.ndarray, Q: np.ndarray, K: int = 100, block_q: int = 1024,
              block_x: int = 1_000_000, candidates: int | None = None,
              use_faiss: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ids [nq,K] int64, dist [nq,K] float32 Euclidean distances).

    Candidates are found with an exhaustive scan (faiss IndexFlatL2 when
    available, else blocked numpy) and reranked in float64 so that near-ties
    are ordered by the exact distance, then by id.
    """
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    nq = Q.shape[0]
    C = max(K, candidates or (2 * K))
    cand_ids = _scan(X, Q, C, block_q, block_x, use_faiss)
    out_ids = np.empty((nq, K), dtype=np.int64)
    out_d = np.empty((nq, K), dtype=np.float32)
    for i in range(nq):
        ids = cand_ids[i]
        ids = ids[ids >= 0]
        v = np.asarray(X[np.sort(ids)], dtype=np.float64)
        ids = np.sort(ids)
        d = np.sqrt(((v - Q[i].astype(np.float64)) ** 2).sum(axis=1))
        order = np.lexsort((ids, d))[:K]
        out_ids[i] = ids[order]
        out_d[i] = d[order].astype(np.float32)
    return out_ids, out_d


def _scan(X, Q, C, block_q, block_x, use_faiss):
    n, d = X.shape
    if use_faiss:
        try:
            import faiss  # type: ignore

            best_d = np.full((Q.shape[0], C), np.inf, dtype=np.float32)
            best_i = np.full((Q.shape[0], C), -1, dtype=np.int64)
            for s in range(0, n, block_x):
                xb = np.ascontiguousarray(X[s:s + block_x], dtype=np.float32)
                index = faiss.IndexFlatL2(d)
                index.add(xb)
                dd, ii = index.search(Q, min(C, xb.shape[0]))
                ii = np.where(ii >= 0, ii + s, -1)
                best_d, best_i = _merge(best_d, best_i, dd, ii, C)
            return best_i
        except ImportError:
            pass
    best_d = np.full((Q.shape[0], C), np.inf, dtype=np.float32)
    best_i = np.full((Q.shape[0], C), -1, dtype=np.int64)
    qn = (Q.astype(np.float32) ** 2).sum(1)
    for s in range(0, n, block_x):
        xb = np.ascontiguousarray(X[s:s + block_x], dtype=np.float32)
        xn = (xb ** 2).sum(1)
        for a in range(0, Q.shape[0], block_q):
            qb = Q[a:a + block_q]
            dd = qn[a:a + block_q, None] + xn[None, :] - 2.0 * (qb @ xb.T)
            c = min(C, xb.shape[0])
            part = np.argpartition(dd, c - 1, axis=1)[:, :c]
            pd = np.take_along_axis(dd, part, axis=1)
            nd, ni = _merge(best_d[a:a + block_q], best_i[a:a + block_q], pd,
                            part.astype(np.int64) + s, C)
            best_d[a:a + block_q], best_i[a:a + block_q] = nd, ni
    return best_i


def _merge(d1, i1, d2, i2, C):
    d = np.concatenate([d1, d2], axis=1)
    i = np.concatenate([i1, i2], axis=1)
    c = min(C, d.shape[1])
    part = np.argpartition(d, c - 1, axis=1)[:, :c]
    return np.take_along_axis(d, part, axis=1), np.take_along_axis(i, part, axis=1)


def recall_at_k(labels: np.ndarray, gt_ids: np.ndarray, k: int) -> np.ndarray:
    """Per-query recall@k (labels: n x >=k, -1 padded)."""
    n = labels.shape[0]
    r = np.empty(n, dtype=np.float64)
    for i in range(n):
        got = labels[i, :k]
        r[i] = np.intersect1d(got[got >= 0], gt_ids[i, :k]).size / k
    return r
