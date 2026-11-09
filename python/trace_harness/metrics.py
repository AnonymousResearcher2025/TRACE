"""Per-run summaries and aggregation across repetitions."""
from __future__ import annotations

import numpy as np

from .gt import recall_at_k


def worst_window(x: np.ndarray, w: int = 100) -> float:
    if len(x) < w:
        return float(np.mean(x)) if len(x) else float("nan")
    c = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    return float(((c[w:] - c[:-w]) / w).min())


def summarize(out: dict, gt_ids: np.ndarray | None, k: int, mask: np.ndarray | None = None,
              device_read_bytes: int | None = None, page: int = 4096) -> dict:
    """Summary of one replay restricted to ``mask`` (e.g. post-warmup)."""
    n_all = len(out["path"])
    m = np.ones(n_all, bool) if mask is None else mask
    n = int(m.sum())
    path = out["path"][m]
    wall = out["wall_ns"][m] / 1e6
    s = {
        "n_queries": n,
        "dist_mean": float(out["n_dist"][m].mean()),
        "pq_mean": float(out["n_pq"][m].mean()),
        "aux_mean": float(out["n_aux"][m].mean()),
        "reads_mean": float(out["n_reads"][m].mean()),
        "cache_hits_mean": float(out["n_cache_hits"][m].mean()),
        "pruned_mean": float(out["n_pruned"][m].mean()),
        "R_mean": float(out["R"][m].mean()),
        "hops_mean": float(out["n_hops"][m].mean()),
        "frac_with_seeds": float((out["n_seeds"][m] > 0).mean()),
        "p50_ms": float(np.percentile(wall, 50)),
        "p95_ms": float(np.percentile(wall, 95)),
        "p99_ms": float(np.percentile(wall, 99)),
        "mean_ms": float(wall.mean()),
        "cpu_p50_ms": float(np.percentile(out["cpu_ns"][m] / 1e6, 50)),
        "cpu_mean_ms": float((out["cpu_ns"][m] / 1e6).mean()),
        "io_p50_ms": float(np.percentile(out["io_ns"][m] / 1e6, 50)),
        "io_mean_ms": float((out["io_ns"][m] / 1e6).mean()),
        "frac_cold": float((path == 0).mean()),
        "frac_resolve": float((path == 1).mean()),
        "frac_seeded": float((path == 2).mean()),
        "frac_cache_hit": float((path == 3).mean()),
        "early_stop_frac": float(out["early_stopped"][m].mean()),
        "audit_checked": int(out["audit_checked"][m].sum()),
        "audit_violations": int(out["audit_viol"][m].sum()),
        "resolve_exact_fail": int((out["resolve_exact"][m] == 0).sum()),
        "prune_equiv_fail": int((out["prune_equiv"][m] == 0).sum()),
        "prune_equiv_checked": int((out["prune_equiv"][m] != -1).sum()),
    }
    if device_read_bytes is not None:
        s["device_reads_mean"] = device_read_bytes / page / max(n_all, 1)
    if gt_ids is not None:
        rec = recall_at_k(out["labels"][m], gt_ids[m], k)
        s.update({
            "recall_mean": float(rec.mean()),
            "recall_p5": float(np.percentile(rec, 5)),
            "recall_full_frac": float((rec == 1.0).mean()),
            "recall_worst100": worst_window(rec, 100),
        })
        if (path == 1).any():
            rr = rec[path == 1]
            ell = out["ell"][m][path == 1]
            s["resolved_recall_mean"] = float(rr.mean())
            s["resolved_recall_by_chain"] = {
                "1": _mean(rr[ell == 1]), "2-4": _mean(rr[(ell >= 2) & (ell <= 4)]),
                "5-8": _mean(rr[(ell >= 5) & (ell <= 8)]), ">8": _mean(rr[ell > 8]),
            }
        if "retained" in out:
            # coverage kappa(q) = |NN_k(q) ∩ R| / k on the resolution path
            idx = np.flatnonzero(m)
            kap = []
            for i in idx:
                if out["path"][i] != 1:
                    continue
                R = out["retained"][i]
                g = gt_ids[i][:k]
                kap.append(np.intersect1d(R.astype(np.int64), g).size / k)
            if kap:
                kap = np.asarray(kap)
                s["coverage_full_frac"] = float((kap == 1.0).mean())
                s["coverage_miss_frac"] = float((kap < 1.0).mean())
    return s


def _mean(x):
    return float(x.mean()) if len(x) else float("nan")


def aggregate(runs: list[dict]) -> dict:
    """Median across repetitions for every numeric field (latency percentiles
    are per run, so the median of per-run p99 is reported, as in the paper)."""
    keys = runs[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in runs if k in r]
        if isinstance(vals[0], (int, float)):
            agg[k] = float(np.median(vals))
        else:
            agg[k] = vals[0]
    agg["n_reps"] = len(runs)
    return agg


def ci95(values: list[float]) -> tuple[float, float]:
    """Mean and half-width of a 95% t-interval."""
    from math import sqrt

    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n < 2:
        return float(v.mean()), float("nan")
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
         9: 2.306, 10: 2.262}.get(n, 1.96)
    return float(v.mean()), float(t * v.std(ddof=1) / sqrt(n))
