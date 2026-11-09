"""Experiment plumbing: tiers, workload splits, tuning, and measured runs."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import trace_harness as th

from .gt import recall_at_k
from .io import Workload, load_json, save_json
from .methods import TRACE_VARIANTS, Ctx, build, trace_config
from .metrics import aggregate, summarize


# ---------------------------------------------------------------- tiers
@dataclass
class Tier:
    name: str
    type: str                 # "memory" | "disk" | "vamana"
    index: str                # hnswlib file, DiskANN prefix, or Vamana graph file
    dim: int
    workload: str             # workload directory (queries, sessions, gt)
    direct_io: bool = True
    cache_nodes: int = 16384
    beam_width: int = 4
    qv_region_model: str | None = None
    darth_model: str | None = None
    extra_workloads: dict = field(default_factory=dict)
    vectors: str | None = None     # vamana: float32 vectors (.fbin) of the graph
    qvcache: dict = field(default_factory=dict)   # QVCacheConfig overrides
    fidelity: dict = field(default_factory=dict)  # fidelity-check settings

    @staticmethod
    def from_dict(name: str, d: dict) -> Tier:
        return Tier(name=name, **d)


_EXEC_CACHE: dict = {}


def executor(tier: Tier, direct_io: bool | None = None, cache_nodes: int | None = None):
    dio = tier.direct_io if direct_io is None else direct_io
    cn = tier.cache_nodes if cache_nodes is None else cache_nodes
    key = (tier.type, tier.index, dio, cn)
    if key not in _EXEC_CACHE:
        if tier.type == "memory":
            _EXEC_CACHE[key] = th.HnswExecutor(tier.index, tier.dim)
        elif tier.type == "disk":
            _EXEC_CACHE[key] = th.DiskExecutor(tier.index, direct_io=dio,
                                               cache_nodes=cn, max_beam_width=16)
        elif tier.type == "vamana":
            assert tier.vectors, f"tier {tier.name}: vamana needs 'vectors'"
            _EXEC_CACHE[key] = th.VamanaExecutor(tier.index, tier.vectors)
        else:
            raise ValueError(tier.type)
    return _EXEC_CACHE[key]


def context(tier: Tier, ex=None, k: int = 10) -> Ctx:
    return Ctx(executor=ex or executor(tier), k=k, beam_width=tier.beam_width,
               qv_region_model=tier.qv_region_model, darth_model=tier.darth_model,
               qv=dict(tier.qvcache))


# --------------------------------------------------------------- splits
@dataclass
class Split:
    tune: Workload
    replay: Workload
    warmup: int  # leading replay queries excluded from statistics


def split(wl: Workload, tune_queries: int, warmup: int) -> Split:
    """Whole sessions go to the tuning prefix until it holds >= tune_queries
    queries; the rest is replayed, its first ``warmup`` queries unscored."""
    s = wl.sessions
    boundaries = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    cut = len(s)
    for b in boundaries:
        if b >= tune_queries:
            cut = int(b)
            break
    tune_idx = np.arange(0, cut)
    rep_idx = np.arange(cut, len(s))
    assert len(rep_idx) > warmup, "workload too small for the requested split"
    assert not set(np.unique(s[tune_idx])) & set(np.unique(s[rep_idx])), \
        "sessions must be contiguous (tuning and replay sessions overlap)"
    return Split(wl.subset(tune_idx), wl.subset(rep_idx), warmup)


# ---------------------------------------------------------------- tuning
def mean_recall(engine, wl: Workload, k: int, end_sessions: bool = True) -> float:
    out = th.replay(engine, np.ascontiguousarray(wl.Q), wl.sessions, k=k,
                    end_sessions=end_sessions)
    return float(recall_at_k(out["labels"], wl.gt_ids, k).mean())


def tune_L(ctx: Ctx, method: str, wl: Workload, target: float, knobs=None,
           lo: int | None = None, hi: int = 4096) -> tuple[int, float]:
    """Smallest L whose mean recall@k on the tuning split reaches target."""
    k = ctx.k
    lo = lo or k
    cache = {}

    def r(L):
        if L not in cache:
            cache[L] = mean_recall(build(ctx, method, L, knobs), wl, k)
        return cache[L]

    L = max(lo, 16)
    while r(L) < target:
        if L >= hi:
            raise RuntimeError(f"{method}: recall {r(L):.4f} < {target} at L={hi}")
        L = min(hi, L * 2)
    a, b = max(lo, L // 2), L
    if r(a) >= target:
        b = a
    while b - a > 1:
        mid = (a + b) // 2
        if r(mid) >= target:
            b = mid
        else:
            a = mid
    return b, r(b)


def tune_continuous(fn, lo: float, hi: float, target: float, increasing: bool,
                    iters: int = 12) -> tuple[float, float]:
    """Bisection on a monotone knob. increasing=True: recall grows with x
    (returns smallest x reaching target); else returns largest x."""
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        rec = fn(mid)
        ok = rec >= target
        if ok:
            best = (mid, rec)
        if ok == increasing:
            hi = mid
        else:
            lo = mid
    if best is None:
        x = hi if increasing else lo
        best = (x, fn(x))
    return best


def trace_min_L(ctx: Ctx, method: str) -> int | None:
    """Smallest list size L for TRACE variants that resolve. A seeded
    traversal must retain more than one chain of resolutions can consume,
    L > s_max + max(m - 1, 1) * k, so that a traversal can end the chain
    (routing and staleness, Sec. III-D). Returns None for other methods."""
    if method not in TRACE_VARIANTS and method != "trace_catapult":
        return None
    c = trace_config(ctx, 0, **TRACE_VARIANTS.get(method, {}))
    if not (c.enabled and c.resolve):
        return None
    return c.s_max + max(c.m - 1, 1) * c.k + 1


def tuned(results_dir: Path, tier: Tier, ctx: Ctx, method: str, sp: Split,
          target: float, base_L: int | None = None) -> dict:
    """Cached tuning: returns {"L": .., "knobs": .., "tune_recall": ..}."""
    path = results_dir / "tuning" / tier.name / f"{method}.json"
    if path.exists():
        return load_json(path)
    k = ctx.k
    if method == "qvcache":
        L = base_L or tune_L(ctx, "cold", sp.tune, target)[0]

        def f(D):
            return mean_recall(build(ctx, "qvcache", L, {"deviation": D}), sp.tune, k)

        D, rec = tune_continuous(f, 0.0, 1.0, target, increasing=False)
        res = {"L": L, "knobs": {"deviation": D}, "tune_recall": rec}
    elif method == "darth":
        L = 2 * (base_L or tune_L(ctx, "cold", sp.tune, target)[0])

        def f(t):
            return mean_recall(build(ctx, "darth", L, {"target": t}), sp.tune, k)

        t, rec = tune_continuous(f, 0.5, 1.0, target, increasing=True)
        res = {"L": L, "knobs": {"target": t}, "tune_recall": rec}
    else:
        L, rec = tune_L(ctx, method, sp.tune, target, lo=trace_min_L(ctx, method))
        res = {"L": L, "knobs": {}, "tune_recall": rec}
    save_json(path, res)
    return res


# ------------------------------------------------------------- measuring
def read_bytes() -> int | None:
    try:
        with open("/proc/self/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def measure(ctx: Ctx, method: str, L: int, knobs: dict, wl: Workload, warmup: int, *,
            reps: int = 5, diag: dict | None = None, end_sessions: bool = True,
            deadlines_ms=(1000.0 / 30.0,), save_queries: str | None = None) -> dict:
    """Replays ``wl`` ``reps`` times with a fresh engine each time and returns
    the median summary (per-run percentiles, as in the paper)."""
    runs = []
    mask = np.zeros(len(wl), bool)
    mask[warmup:] = True
    keep = bool(diag and diag.get("audit"))
    for rep in range(reps):
        eng = build(ctx, method, L, knobs, diag)
        rb0 = read_bytes()
        t0 = time.perf_counter()
        out = th.replay(eng, np.ascontiguousarray(wl.Q), wl.sessions, k=ctx.k,
                        end_sessions=end_sessions, keep_retained=keep)
        elapsed = time.perf_counter() - t0
        rb1 = read_bytes()
        dev = None if rb0 is None or rb1 is None else rb1 - rb0
        s = summarize(out, wl.gt_ids, ctx.k, mask, device_read_bytes=dev)
        wall = out["wall_ns"][mask] / 1e6
        for dl in deadlines_ms:
            s[f"deadline_miss_{dl:.1f}ms"] = float((wall > dl).mean())
        s["replay_seconds"] = elapsed
        s["state_bytes_end"] = int(eng.state_bytes())
        runs.append(s)
        if save_queries and rep == 0:
            np.savez_compressed(save_queries, **{k_: v for k_, v in out.items()
                                                 if isinstance(v, np.ndarray)})
    agg = aggregate(runs)
    agg.update({"method": method, "L": L, "knobs": knobs})
    return agg


def write_result(results_dir: Path, exp: str, name: str, obj) -> Path:
    p = results_dir / exp / f"{name}.json"
    save_json(p, obj)
    return p


def env_info() -> dict:
    import platform

    info = {"host": platform.node(), "machine": platform.machine(),
            "python": platform.python_version(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            info["cgroup_memory_max"] = f.read().strip()
    except OSError:
        pass
    info["nproc"] = os.cpu_count()
    return info


def load_config(path: str) -> dict:
    import yaml  # type: ignore

    with open(path) as f:
        cfg = yaml.safe_load(f)
    root = cfg.get("data_root", "")

    def sub(v):
        if isinstance(v, str):
            return v.replace("${data_root}", root)
        if isinstance(v, dict):
            return {k: sub(x) for k, x in v.items()}
        if isinstance(v, list):
            return [sub(x) for x in v]
        return v

    return sub(cfg)


__all__ = ["Tier", "executor", "context", "split", "tune_L", "tuned", "measure",
           "write_result", "env_info", "load_config"]
