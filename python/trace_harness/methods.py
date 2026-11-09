"""Method registry: builds an Engine for each row of the paper's tables.

Every method takes the same executor and the tuned search parameter ``L``
(candidate-list capacity). Extra knobs (QVCache deviation, DARTH target) are
passed through ``knobs``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import trace_harness as th

INF = float("inf")


@dataclass
class Ctx:
    """Shared state for building engines on one tier."""
    executor: object
    k: int = 10
    beam_width: int = 4
    max_norm: float = 1.001
    qv_region_model: str | None = None
    qv: dict = field(default_factory=dict)   # QVCacheConfig overrides (config file)
    darth_model: str | None = None
    darth_interval: int = 64
    dcache_budget_bytes: int = 120_000
    catapult_bits: int = 8
    catapult_cap: int = 40
    catapult_seed: int = 42
    extra: dict = field(default_factory=dict)


def trace_config(ctx: Ctx, L: int, **over) -> th.TraceConfig:
    c = th.TraceConfig()
    c.k = ctx.k
    c.L = int(L)
    c.beam_width = ctx.beam_width
    c.max_norm = ctx.max_norm
    # default operating point (Sec. V-A, Methodology)
    c.m, c.tau, c.w, c.s_max, c.B_r = 4, 0.6, 32, 32, 2300
    c.gamma_s, c.L_max = INF, 2**31 - 1
    for k, v in over.items():
        if k == "retention":
            v = getattr(th.TraceConfig.Retention, v)
        if k == "seeds":
            v = getattr(th.TraceConfig.Seeds, v)
        setattr(c, k, v)
    return c


# name -> TraceConfig overrides (all TRACE-family rows and ablations)
TRACE_VARIANTS = {
    "cold": dict(enabled=False),
    "trace": dict(),                                   # TRACE-fast
    "trace_safe": dict(gamma_s=1.5, L_max=8),          # TRACE-safe
    "trace_hybrid": dict(retention="Hybrid"),
    # ablations (Table IV), cumulative
    "warm": dict(transfer=False, seeds="PrevAnswer", prune=False, resolve=False, jump=False),
    "abl_bound_seeds": dict(prune=False, resolve=False, jump=False),
    "abl_prune": dict(resolve=False, jump=False),
    "abl_resolve": dict(jump=False),
    # sensitivity (Table VIII)
    "m1": dict(m=1), "m2": dict(m=2), "m8": dict(m=8),
    "tau04": dict(tau=0.4), "tau08": dict(tau=0.8),
    "w8": dict(w=8), "w128": dict(w=128),
    # session isolation (Table VII, interleaved stream)
    "trace_shared_ring": dict(share_ring=True),
    "trace_hybrid_shared_ring": dict(share_ring=True, retention="Hybrid"),
}


def build(ctx: Ctx, method: str, L: int, knobs: dict | None = None,
          diagnostics: dict | None = None):
    """Returns an Engine. ``diagnostics`` may set audit / verify_prune."""
    knobs = knobs or {}
    diag = diagnostics or {}
    ex = ctx.executor
    if method in TRACE_VARIANTS:
        cfg = trace_config(ctx, L, **TRACE_VARIANTS[method], **diag)
        return th.TraceEngine(ex, cfg)
    if method in ("catapult", "trace_catapult"):
        cat = th.Catapult(ex.dim, ctx.catapult_bits, ctx.catapult_cap, ctx.catapult_seed)
        over = dict(enabled=False) if method == "catapult" else {}
        cfg = trace_config(ctx, L, **over, **diag)
        return th.TraceEngine(ex, cfg, cat)
    if method == "darth":
        assert ctx.darth_model, "DARTH needs a trained model (scripts/train_darth.py)"
        stop = th.DarthStop(ctx.darth_model, float(knobs.get("target", 0.95)),
                            int(knobs.get("interval", ctx.darth_interval)))
        cfg = trace_config(ctx, L, enabled=False)
        return th.TraceEngine(ex, cfg, None, stop)
    if method == "qvcache":
        assert ctx.qv_region_model, "QVCache needs a region model (scripts/fit_qvcache.py)"
        c = th.QVCacheConfig()  # defaults follow the QVCache paper (baselines.h)
        c.k, c.L, c.beam_width = ctx.k, int(L), ctx.beam_width
        for key, v in {**ctx.qv, **knobs}.items():
            if key == "strategy":
                v = getattr(th.QVStrategy, v)
            if not hasattr(c, key):
                raise KeyError(f"unknown QVCache option {key!r}")
            setattr(c, key, v)
        c.region_model = ctx.qv_region_model
        return th.QVCacheEngine(ex, c)
    if method == "dcache":
        c = th.DCacheConfig()
        c.k, c.L, c.beam_width, c.max_norm = ctx.k, int(L), ctx.beam_width, ctx.max_norm
        c.pivots = int(knobs.get("pivots", 4))
        c.budget_bytes = int(knobs.get("budget_bytes", ctx.dcache_budget_bytes))
        return th.DCacheEngine(ex, c)
    if method == "snake":
        c = th.SnakeConfig()
        c.k, c.L, c.beam_width, c.max_norm = ctx.k, int(L), ctx.beam_width, ctx.max_norm
        c.slots = int(knobs.get("slots", 4))
        return th.SnakeEngine(ex, c)
    raise KeyError(f"unknown method {method!r}")


ALL_METHODS = sorted(list(TRACE_VARIANTS) + ["catapult", "trace_catapult", "darth",
                                             "qvcache", "dcache", "snake"])

# Methods whose state spans sessions (must see the stream in order, one engine)
GLOBAL_STATE = {"catapult", "trace_catapult", "qvcache", "dcache"}


def is_finite(x: float) -> bool:
    return not (math.isinf(x) or math.isnan(x))
