"""Derived query streams (Sec. V-A workloads c and d).

All functions take and return (Q, sessions) arrays; queries are unit-norm
float32. ``knn_scale`` computes scale_k (distance to the k-th nearest corpus
vector) exactly, which the controlled-locality generator needs at each step.
"""
from __future__ import annotations

import numpy as np

from .gt import exact_knn
from .io import l2_normalize


def shuffled(Q: np.ndarray, seed: int = 0):
    """One session with the stream order randomly permuted (no locality)."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(Q))
    return np.ascontiguousarray(Q[p]), np.zeros(len(Q), dtype=np.int64)


def _sessions(S):
    starts = np.flatnonzero(np.r_[True, S[1:] != S[:-1]])
    ends = np.r_[starts[1:], len(S)]
    return [(a, b) for a, b in zip(starts, ends)]


def alternating(Q, S, block: int = 16, seed: int = 0, max_sessions: int | None = None):
    """Pairs of clips interleaved in blocks of ``block`` queries within one
    session (A B A B ...), so the stream revisits a region every 2*block."""
    rng = np.random.default_rng(seed)
    ses = _sessions(S)
    rng.shuffle(ses)
    out_q, out_s = [], []
    sid = 0
    for i in range(0, len(ses) - 1, 2):
        (a0, a1), (b0, b1) = ses[i], ses[i + 1]
        A, B = Q[a0:a1], Q[b0:b1]
        n = min(len(A), len(B))
        for j in range(0, n, block):
            for part in (A[j:j + block], B[j:j + block]):
                out_q.append(part)
                out_s.append(np.full(len(part), sid))
        sid += 1
        if max_sessions and sid >= max_sessions:
            break
    return np.ascontiguousarray(np.concatenate(out_q)), np.concatenate(out_s).astype(np.int64)


def abrupt(Q, S, clips_per_session: int = 10):
    """Consecutive clips concatenated into one session (scene cuts)."""
    ses = _sessions(S)
    sid = np.empty(len(S), dtype=np.int64)
    for i, (a, b) in enumerate(ses):
        sid[a:b] = i // clips_per_session
    return np.ascontiguousarray(Q), sid


def interleaved(Q, S, n: int = 8):
    """Groups of n sessions served round-robin by one executor; session ids
    are preserved (per-session isolation is what is being tested)."""
    ses = _sessions(S)
    out = []
    for g in range(0, len(ses), n):
        grp = ses[g:g + n]
        L = max(b - a for a, b in grp)
        for t in range(L):
            for a, b in grp:
                if a + t < b:
                    out.append(a + t)
    idx = np.asarray(out)
    return np.ascontiguousarray(Q[idx]), S[idx].astype(np.int64)


def knn_scale(X, Q, k: int = 10) -> np.ndarray:
    _, d = exact_knn(X, Q, K=k)
    return d[:, k - 1]


def _move(q, z, target):
    """Bisection on alpha so that ||norm((1-a) q + a z) - q|| = target."""
    lo, hi = 0.0, 1.0

    def step(a):
        v = (1 - a) * q + a * z
        v = v / np.linalg.norm(v)
        return v, np.linalg.norm(v - q)

    v_hi, d_hi = step(hi)
    if d_hi < target:  # cannot reach: take the far end
        return v_hi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        v, d = step(mid)
        if d < target:
            lo = mid
        else:
            hi = mid
    return step(hi)[0]


def controlled_locality(X, seeds_Q, rho_lo: float, rho_hi: float, n_queries: int,
                        chain_len: int = 50, k: int = 10, seed: int = 0):
    """Chains that start at real queries; each step moves to a random
    direction by rho * scale_k(q_t), rho ~ U[rho_lo, rho_hi). Chains advance in
    lockstep so scale_k is computed in batches."""
    rng = np.random.default_rng(seed)
    n_chains = int(np.ceil(n_queries / chain_len))
    starts = seeds_Q[rng.choice(len(seeds_Q), n_chains, replace=len(seeds_Q) < n_chains)]
    cur = np.asarray(starts, dtype=np.float64)
    Qs = np.empty((n_chains, chain_len, X.shape[1]), dtype=np.float32)
    for t in range(chain_len):
        Qs[:, t] = l2_normalize(cur.astype(np.float32))
        if t + 1 == chain_len:
            break
        sc = knn_scale(X, Qs[:, t], k)
        nxt = np.empty_like(cur)
        for c in range(n_chains):
            q = Qs[c, t].astype(np.float64)
            z = rng.standard_normal(X.shape[1])
            z /= np.linalg.norm(z)
            rho = rng.uniform(rho_lo, rho_hi)
            nxt[c] = _move(q, z, rho * float(sc[c]))
        cur = nxt
    Q = Qs.reshape(-1, X.shape[1])[:n_queries]
    S = np.repeat(np.arange(n_chains), chain_len)[:n_queries].astype(np.int64)
    return np.ascontiguousarray(Q), S


def oscillating(X, seeds_Q, tau: float, n_queries: int, chain_len: int = 50,
                eps: float = 0.1, k: int = 10, seed: int = 0):
    """Random-direction walk whose step alternates between rho = tau - eps
    and tau + eps, so the router flips between transfer and fallback."""
    rng = np.random.default_rng(seed)
    n_chains = int(np.ceil(n_queries / chain_len))
    cur = np.asarray(seeds_Q[rng.choice(len(seeds_Q), n_chains,
                                        replace=len(seeds_Q) < n_chains)], dtype=np.float64)
    Qs = np.empty((n_chains, chain_len, X.shape[1]), dtype=np.float32)
    for t in range(chain_len):
        Qs[:, t] = l2_normalize(cur.astype(np.float32))
        if t + 1 == chain_len:
            break
        sc = knn_scale(X, Qs[:, t], k)
        rho = tau - eps if t % 2 == 0 else tau + eps
        for c in range(n_chains):
            z = rng.standard_normal(X.shape[1])
            z /= np.linalg.norm(z)
            cur[c] = _move(Qs[c, t].astype(np.float64), z, rho * float(sc[c]))
    Q = Qs.reshape(-1, X.shape[1])[:n_queries]
    S = np.repeat(np.arange(n_chains), chain_len)[:n_queries].astype(np.int64)
    return np.ascontiguousarray(Q), S


def locality_ratio(Q, S, scale_k):
    """rho_t = ||q_{t+1} - q_t|| / scale_k(q_t) within sessions."""
    same = S[1:] == S[:-1]
    d = np.linalg.norm(Q[1:].astype(np.float64) - Q[:-1], axis=1)
    rho = d / np.maximum(scale_k[:-1], 1e-12)
    return rho[same]
