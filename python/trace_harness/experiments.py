"""Experiment definitions, one per table of the paper's evaluation (Sec. V).

Each experiment writes one JSON summary per tier to results/<exp>/<tier>.json.
scripts/make_tables.py prints the paper's tables from these files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import trace_harness as th

from .gt import recall_at_k
from .io import Workload
from .metrics import ci95
from .runner import (Tier, context, env_info, executor, measure, split, tuned,
                     write_result)

BASELINE_METHODS = ["cold", "darth", "qvcache", "dcache", "snake", "catapult",
                    "warm", "trace", "trace_catapult"]
ATTRIBUTION_METHODS = ["cold", "warm", "abl_bound_seeds", "abl_prune", "abl_resolve",
                       "trace", "m1"]
SENSITIVITY_METHODS = ["m1", "m2", "trace", "m8", "tau04", "tau08", "w8", "w128",
                       "trace_safe"]


class Runner:
    def __init__(self, cfg: dict, results: str, reps: int | None = None,
                 limit: int | None = None):
        self.cfg = cfg
        self.results = Path(results)
        self.k = int(cfg.get("k", 10))
        self.target = float(cfg.get("target_recall", 0.95))
        self.reps = int(reps or cfg.get("reps", 5))
        self.warmup = int(cfg.get("warmup", 2000))
        self.tune_queries = int(cfg.get("tune_queries", 5000))
        self.limit = limit
        self.tiers = {n: Tier.from_dict(n, d) for n, d in cfg["tiers"].items()}

    # -- helpers
    def _split(self, tier: Tier, workload: str | None = None):
        wl = Workload(workload or tier.workload)
        assert wl.gt_ids is not None, f"ground truth missing for {wl.root}"
        sp = split(wl, self.tune_queries, self.warmup)
        if self.limit:
            n = min(len(sp.replay), self.warmup + self.limit)
            sp.replay = sp.replay.subset(np.arange(n))
        return sp

    def _tuned(self, tier, ctx, method, sp):
        base = None
        if method in ("qvcache", "darth"):
            base = tuned(self.results, tier, ctx, "cold", sp, self.target)["L"]
        return tuned(self.results, tier, ctx, method, sp, self.target, base_L=base)

    def _measure(self, ctx, method, L, knobs, wl, warmup=None, **kw):
        w = self.warmup if warmup is None else warmup
        return measure(ctx, method, L, knobs, wl, w, reps=self.reps, **kw)

    def _save(self, exp, name, obj):
        obj = dict(obj)
        obj["env"] = env_info()
        p = write_result(self.results, exp, name, obj)
        print(f"[{exp}] wrote {p}")

    # ------------------------------------- Table II: cost at matched recall
    def cost(self, tiers):
        for tn in tiers:
            tier = self.tiers[tn]
            ctx = context(tier, k=self.k)
            sp = self._split(tier)
            res = {}
            for m in ("cold", "trace"):
                t = self._tuned(tier, ctx, m, sp)
                res[m] = self._measure(ctx, m, t["L"], t["knobs"], sp.replay)
            # 5 disjoint stream samples -> 95% CI of the gain
            res["ci"] = self._samples(tier, ctx, sp, res)
            self._save("cost", tn, res)

    def _samples(self, tier, ctx, sp, res, n_groups: int = 5):
        sess = np.unique(sp.replay.sessions)
        groups = np.array_split(sess, n_groups)
        key = "reads_mean" if tier.type == "disk" else "dist_mean"
        gains, lat = [], []
        for g in groups:
            idx = np.flatnonzero(np.isin(sp.replay.sessions, g))
            wl = sp.replay.subset(idx)
            warm = min(self.warmup // n_groups, len(wl) // 4)
            c = measure(ctx, "cold", res["cold"]["L"], {}, wl, warm, reps=1)
            t = measure(ctx, "trace", res["trace"]["L"], {}, wl, warm, reps=1)
            gains.append(c[key] / max(t[key], 1e-9))
            lat.append(c["p50_ms"] / max(t["p50_ms"], 1e-9))
        mg, hg = ci95(gains)
        ml, hl = ci95(lat)
        return {"metric": key, "gain_mean": mg, "gain_ci95": hg,
                "p50_gain_mean": ml, "p50_gain_ci95": hl, "groups": len(groups)}

    # -------------------------------------------- Table III: baselines
    def baselines(self, tiers, methods=None):
        for tn in tiers:
            tier = self.tiers[tn]
            ctx = context(tier, k=self.k)
            sp = self._split(tier)
            out = {}
            for m in methods or BASELINE_METHODS:
                if m == "darth" and not tier.darth_model:
                    print(f"skip darth on {tn}: no model")
                    continue
                if m == "qvcache" and not tier.qv_region_model:
                    print(f"skip qvcache on {tn}: no region model")
                    continue
                t = self._tuned(tier, ctx, m, sp)
                out[m] = self._measure(ctx, m, t["L"], t["knobs"], sp.replay)
                out[m]["tune_recall"] = t["tune_recall"]
                self._save("baselines", tn, out)

    # ------------------------------------------ Table IV: attribution
    def attribution(self, tiers):
        for tn in tiers:
            tier = self.tiers[tn]
            ctx = context(tier, k=self.k)
            sp = self._split(tier)
            # every row is tuned to the target, as in Table III; m=1 keeps TRACE's L
            Lt = self._tuned(tier, ctx, "trace", sp)["L"]
            out = {}
            for m in ATTRIBUTION_METHODS:
                Lm = Lt if m == "m1" else self._tuned(tier, ctx, m, sp)["L"]
                out[m] = self._measure(ctx, m, Lm, {}, sp.replay)
            if tier.type != "disk":  # Theorem 2 is an in-memory guarantee
                out["same_seeds"] = _same_seed_pruning(ctx, Lt, sp.replay, self.warmup)
            self._save("attribution", tn, out)

    # ------------------------------- Table V: storage and the page cache
    def storage(self, tiers, label: str):
        """Run once per memory cap / cache configuration. ``label`` names the
        configuration: direct_16k, direct_64k, buffered_1g, buffered_2g, or
        buffered_nocap. Labels starting with "buffered" use buffered reads, and
        "64k" selects a 65,536-node cache. Run capped configurations inside a
        cgroup with the corresponding memory.max (scripts/run_all.sh)."""
        for tn in tiers:
            tier = self.tiers[tn]
            assert tier.type == "disk"
            sp = self._split(tier)
            base_ctx = context(tier, k=self.k)
            Lc = self._tuned(tier, base_ctx, "cold", sp)["L"]
            Lt = self._tuned(tier, base_ctx, "trace", sp)["L"]
            direct = not label.startswith("buffered")
            cache = 65536 if "64k" in label else tier.cache_nodes
            ex = executor(tier, direct_io=direct, cache_nodes=cache)
            ctx = context(tier, ex=ex, k=self.k)
            out = {"label": label, "direct_io": direct, "cache_nodes": cache,
                   "cold": self._measure(ctx, "cold", Lc, {}, sp.replay),
                   "trace": self._measure(ctx, "trace", Lt, {}, sp.replay)}
            self._save("storage", f"{tn}_{label}", out)

    # -------------------------------------------- Table VI: locality sweep
    def locality(self, tiers):
        for tn in tiers:
            tier = self.tiers[tn]
            ctx = context(tier, k=self.k)
            L = self._tuned(tier, ctx, "cold", self._split(tier))["L"]
            out = {"L": L}
            for name, path in tier.extra_workloads.get("locality", {}).items():
                wl = Workload(path)
                warm = min(self.warmup, len(wl) // 10)
                out[name] = {m: measure(ctx, m, L, {}, wl, warm, reps=self.reps)
                             for m in ("cold", "trace")}
            self._save("locality", tn, out)

    # ------------------------------------------ Table VII: stream dynamics
    def dynamics(self, tiers):
        for tn in tiers:
            tier = self.tiers[tn]
            ctx = context(tier, k=self.k)
            L = self._tuned(tier, ctx, "cold", self._split(tier))["L"]
            out = {"L": L}
            for name, path in tier.extra_workloads.get("dynamics", {}).items():
                wl = Workload(path)
                warm = min(self.warmup, len(wl) // 10)
                interleaved = name.startswith("interleaved")
                ms = ["cold", "trace", "trace_hybrid"]
                if interleaved:
                    ms += ["trace_shared_ring", "trace_hybrid_shared_ring"]
                out[name] = {m: measure(ctx, m, L, {}, wl, warm, reps=self.reps,
                                        end_sessions=not interleaved) for m in ms}
            self._save("dynamics", tn, out)

    # --------------------------------------------- Table VIII: sensitivity
    def sensitivity(self, tiers):
        for tn in tiers:
            tier = self.tiers[tn]
            ctx = context(tier, k=self.k)
            sp = self._split(tier)
            L = self._tuned(tier, ctx, "cold", sp)["L"]
            out = {"L": L, "cold": self._measure(ctx, "cold", L, {}, sp.replay)}
            for m in SENSITIVITY_METHODS:
                # audit=True exports R so the coverage-miss rate can be computed
                out[m] = measure(ctx, m, L, {}, sp.replay, self.warmup, reps=1,
                                 diag={"audit": True})
                timing = self._measure(ctx, m, L, {}, sp.replay)
                for key in ("p50_ms", "p95_ms", "p99_ms", "mean_ms"):
                    out[m][key] = timing[key]
            self._save("sensitivity", tn, out)

    # ------------------------------------ Sec. V-G: guarantees in practice
    def guarantees(self, tiers, audit_queries: int = 20000):
        for tn in tiers:
            tier = self.tiers[tn]
            ctx = context(tier, k=self.k)
            sp = self._split(tier)
            out = {}
            for m in ("trace", "trace_safe"):
                t = self._tuned(tier, ctx, m, sp)
                n = min(len(sp.replay), self.warmup + audit_queries)
                wl = sp.replay.subset(np.arange(n))
                out[m] = measure(ctx, m, t["L"], {}, wl, self.warmup, reps=1,
                                 diag={"audit": True, "verify_prune": tier.type == "memory"})
            out["cold"] = self._measure(ctx, "cold", self._tuned(tier, ctx, "cold", sp)["L"],
                                        {}, sp.replay)
            self._save("guarantees", tn, out)

    # ------------------------------------ Table X (top): concurrent sessions
    def scale(self, tiers, sessions=(1, 16, 256), threads: int = 16,
              queries_per_session: int = 150):
        for tn in tiers:
            tier = self.tiers[tn]
            ctx = context(tier, k=self.k)
            sp = self._split(tier)
            out = {"threads": threads}
            for m in ("cold", "trace"):
                L = self._tuned(tier, ctx, m, sp)["L"]
                for S in sessions:
                    wl = _concurrent_sessions(sp.replay, S, queries_per_session)
                    engines = [_build(ctx, m, L) for _ in range(min(threads, S))]
                    res = []
                    for _ in range(self.reps):
                        for e in engines:
                            e.reset()
                        o = th.replay_parallel(engines, np.ascontiguousarray(wl.Q),
                                               wl.sessions, k=self.k, end_sessions=False)
                        res.append({"qps": len(wl) / (o["total_wall_ns"] / 1e9),
                                    "state_bytes": int(sum(e.state_bytes() for e in engines))})
                    out[f"{m}_S{S}"] = {"qps": float(np.median([r["qps"] for r in res])),
                                        "state_bytes": res[-1]["state_bytes"], "L": L,
                                        "engines": len(engines)}
            self._save("scale", tn, out)

    # ------------------------------------- Table X (bottom): embedding width
    def width(self, tiers):
        for tn in tiers:
            tier = self.tiers[tn]
            ctx = context(tier, k=self.k)
            sp = self._split(tier)
            out = {}
            for m in ("cold", "trace"):
                t = self._tuned(tier, ctx, m, sp)
                out[m] = self._measure(ctx, m, t["L"], {}, sp.replay)
            # steady-state ring payload per session
            t = self._tuned(tier, ctx, "trace", sp)
            out["ring_bytes_per_session"] = _ring_bytes(ctx, t["L"], sp.replay)
            self._save("width", tn, out)

    # ------------------- Sec. V-A: fidelity of the reimplemented baselines
    def fidelity(self, tiers):
        """Reproduces each baseline paper's own workload (fidelity.py) and
        stores our measurements next to the values the paper reports."""
        from .fidelity import CATAPULT_REPORTED, QVCACHE_REPORTED

        for tn in tiers:
            tier = self.tiers[tn]
            f = dict(tier.fidelity)
            wl = Workload(tier.workload)
            assert wl.gt_ids is not None, f"ground truth missing for {wl.root}"
            if self.limit:
                wl = wl.subset(np.arange(min(len(wl), self.limit)))
            if f.get("kind") == "qvcache":
                out = self._fid_qvcache(tier, f, wl)
                out["reported"] = QVCACHE_REPORTED
            elif f.get("kind") == "catapult":
                out = self._fid_catapult(tier, f, wl)
                out["reported"] = {str(k): v for k, v in CATAPULT_REPORTED.items()}
            else:
                raise ValueError(f"tier {tn}: fidelity.kind must be qvcache or catapult")
            self._save("fidelity", tn, out)

    def _fid_qvcache(self, tier, f, wl):
        from .fidelity import per_group, shift_drop_pct
        from .gt import recall_at_k
        from .methods import build
        from .runner import tune_L

        k = int(f.get("k", self.k))
        ctx = context(tier, k=k)
        if f.get("L") is not None:
            L = int(f["L"])
        else:  # backend list size at the target recall, on a prefix of the stream
            pre = wl.subset(np.arange(min(len(wl), self.tune_queries)))
            L = tune_L(ctx, "cold", pre, self.target)[0]
        knobs = {"deviation": float(f.get("deviation", 0.075))}
        disp = np.load(Path(wl.root) / "dispatch.npy")[: len(wl)]
        n_repeat = int(wl.meta.get("n_repeat", 3))
        res = {"k": k, "L": L, "knobs": knobs, "workload": str(wl.root),
               "n_queries": len(wl)}
        for m in ("cold", "qvcache"):
            res[m] = self._measure(ctx, m, L, knobs if m == "qvcache" else {}, wl, warmup=0)
        curves = {}
        for m in ("cold", "qvcache"):
            eng = build(ctx, m, L, knobs if m == "qvcache" else {})
            o = th.replay(eng, np.ascontiguousarray(wl.Q), wl.sessions, k=k)
            c = {"recall_by_dispatch": per_group(recall_at_k(o["labels"], wl.gt_ids, k), disp)}
            if m == "qvcache":
                hit = o["path"] == 3
                c["hit_ratio_by_dispatch"] = per_group(hit.astype(np.float64), disp)
                c["hit_latency_p50_ms"] = (float(np.percentile(o["wall_ns"][hit] / 1e6, 50))
                                           if hit.any() else float("nan"))
                c["cached_vectors_end"] = int(eng.cached_vectors)
            curves[m] = c
        res["curves"] = curves
        cold, qv = res["cold"], res["qvcache"]
        res["summary"] = {
            "hit_ratio": qv["frac_cache_hit"],
            "recall_cold": cold["recall_mean"],
            "recall_qvcache": qv["recall_mean"],
            "recall_drop_points": 100.0 * (cold["recall_mean"] - qv["recall_mean"]),
            "recall_drop_pct": 100.0 * (cold["recall_mean"] - qv["recall_mean"])
                               / max(cold["recall_mean"], 1e-12),
            "hit_ratio_drop_at_shift_pct": shift_drop_pct(
                curves["qvcache"]["hit_ratio_by_dispatch"], n_repeat),
            "hit_latency_p50_ms": curves["qvcache"]["hit_latency_p50_ms"],
            "p50_reduction_x": cold["p50_ms"] / max(qv["p50_ms"], 1e-9),
            "state_bytes_end": qv["state_bytes_end"],
        }
        return res

    def _fid_catapult(self, tier, f, wl):
        out = {"workload": str(wl.root), "n_queries": len(wl)}
        for kk in f.get("ks", [1, 16]):
            kk = int(kk)
            ctx = context(tier, k=kk)
            # CatapultDB's k is both the result count and the beam (list) size
            L = kk if f.get("L_equals_k", True) else int(f["L"])
            r = {m: self._measure(ctx, m, L, {}, wl, warmup=0) for m in ("cold", "catapult")}
            c, a = r["cold"], r["catapult"]
            r["summary"] = {
                "L": L,
                "hops_reduction_pct": 100.0 * (1.0 - a["hops_mean"] / max(c["hops_mean"], 1e-12)),
                "dist_reduction_pct": 100.0 * (1.0 - a["dist_mean"] / max(c["dist_mean"], 1e-12)),
                "usage_pct": 100.0 * a["frac_with_seeds"],
                "recall_cold": c["recall_mean"],
                "recall_catapult": a["recall_mean"],
                "mean_ms_cold": c["mean_ms"],
                "mean_ms_catapult": a["mean_ms"],
            }
            out[str(kk)] = r
        return out


def _same_seed_pruning(ctx, L, wl: Workload, warmup: int) -> dict:
    """Pruning with the seeds held fixed (Sec. V-D). One TRACE replay in which
    every seeded-path search is rerun from the same seeds without pruning,
    outside the timed region (verify_prune). Reports per-query distances and
    p50 latency with and without pruning on the scored seeded-path queries,
    and whether all result lists are identical (Theorem 2)."""
    from .methods import build

    eng = build(ctx, "trace", L, diagnostics={"verify_prune": True})
    o = th.replay(eng, np.ascontiguousarray(wl.Q), wl.sessions, k=ctx.k)
    scored = np.arange(len(wl)) >= warmup
    seeded = scored & (o["path"] == 2) & (o["prune_equiv"] != -1)
    if not seeded.any():
        return {"n_seeded": 0}
    eq = o["prune_equiv"][seeded]
    rec = recall_at_k(o["labels"][seeded], wl.gt_ids[seeded], ctx.k)
    return {
        "n_seeded": int(seeded.sum()),
        "dist_prune": float(o["n_dist"][seeded].mean()),
        "dist_noprune": float(o["noprune_n_dist"][seeded].mean()),
        "p50_ms_prune": float(np.percentile(o["wall_ns"][seeded] / 1e6, 50)),
        "p50_ms_noprune": float(np.percentile(o["noprune_wall_ns"][seeded] / 1e6, 50)),
        "recall_mean": float(rec.mean()),
        "prune_equiv_fail": int((eq == 0).sum()),
        "identical_results": bool((eq == 1).all()),
    }


def _build(ctx, m, L):
    from .methods import build

    return build(ctx, m, L)


def _concurrent_sessions(wl: Workload, S: int, per: int) -> Workload:
    """S concurrent sessions of ``per`` consecutive queries each, interleaved
    round-robin. Long recordings are cut into consecutive segments, each a
    separate session (a video clip yields several sessions)."""
    s = wl.sessions
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    ends = np.r_[starts[1:], len(s)]
    while True:
        segs = [np.arange(a + j, a + j + per) for a, b in zip(starts, ends)
                for j in range(0, b - a - per + 1, per)]
        if len(segs) >= S or per <= 20:
            break
        per = max(20, per // 2)
    assert len(segs) >= S, f"workload too small for {S} sessions"
    order = np.stack(segs[:S], axis=1).reshape(-1)
    sub = wl.subset(order)
    sub.sessions = np.repeat(np.arange(S, dtype=np.int64)[None, :], per, axis=0).reshape(-1)
    return sub


def _ring_bytes(ctx, L, wl: Workload) -> float:
    eng = _build(ctx, "trace", L)
    th.replay(eng, np.ascontiguousarray(wl.Q), wl.sessions, k=ctx.k, end_sessions=False)
    return eng.state_bytes() / max(len(np.unique(wl.sessions)), 1)
