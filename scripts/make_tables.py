#!/usr/bin/env python3
"""Prints the paper's evaluation tables (Tables II-X), the guarantee audit of
Sec. V-G, and the baseline fidelity checks of Sec. V-A from results/*.json.

  python scripts/make_tables.py --results results

Tables whose result files are missing are skipped. Counts are means per scored
query; latencies are medians across repetitions of per-run percentiles (ms).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

METHOD_NAMES = {
    "cold": "Cold search",
    "darth": "DARTH-style",
    "qvcache": "QVCache",
    "dcache": "D-Cache",
    "snake": "Snake Table",
    "catapult": "CatapultDB",
    "warm": "Warm seeds",
    "trace": "TRACE",
    "trace_catapult": "TRACE+CatapultDB",
}
ATTRIBUTION_ROWS = [
    ("cold", "Cold search"),
    ("warm", "+ previous-answer seeds"),
    ("abl_bound_seeds", "+ bound-selected seeds (m=4)"),
    ("abl_prune", "+ interval pruning"),
    ("abl_resolve", "+ traversal-free resolution"),
    ("trace", "+ jump detection (TRACE)"),
    ("m1", "TRACE with m=1"),
]
STORAGE_ROWS = [
    ("direct_16k", "O_DIRECT, 16K-node cache, 2 GiB"),
    ("direct_64k", "O_DIRECT, 64K-node cache, 2 GiB"),
    ("buffered_2g", "buffered I/O, 2 GiB"),
    ("buffered_1g", "buffered I/O, 1 GiB"),
    ("buffered_nocap", "buffered I/O, no cap"),
]
LOCALITY_ROWS = [
    ("rho_0_0.125", "[0, .125)"),
    ("rho_0.125_0.25", "[.125, .25)"),
    ("rho_0.25_0.5", "[.25, .5)"),
    ("rho_0.5_1", "[.5, 1)"),
    ("rho_1_2", "[1, 2)"),
    ("shuffled", "shuffled"),
]
DYNAMICS_ROWS = [
    ("natural", "natural", "trace", "trace_hybrid"),
    ("alternating", "alternating", "trace", "trace_hybrid"),
    ("abrupt", "abrupt", "trace", "trace_hybrid"),
    ("oscillating", "oscillating", "trace", "trace_hybrid"),
    ("interleaved", "interleaved, per-session", "trace", "trace_hybrid"),
    ("interleaved", "interleaved, one ring", "trace_shared_ring", "trace_hybrid_shared_ring"),
]
SENSITIVITY_ROWS = [
    ("cold", "cold search"),
    ("m1", "m=1"),
    ("m2", "m=2"),
    ("trace", "default (m=4, tau=0.6, w=32)"),
    ("m8", "m=8"),
    ("tau04", "tau=0.4"),
    ("tau08", "tau=0.8"),
    ("w8", "w=8"),
    ("w128", "w=128"),
    ("trace_safe", "TRACE-safe (gamma_s=1.5, L_max=8)"),
]
WIDTH_ROWS = [
    ("mem_dialog", "384-d (MiniLM-L6-v2)"),
    ("mem_dialog_768", "768-d (MPNet)"),
    ("mem_dialog_1536", "1536-d (OpenAI)"),
]
NA = "n/a"
# TRACE's default operating point (Sec. V-A): anchors, result size, straddler cap.
M_ANCHORS, K_RESULT, S_MAX = 4, 10, 32


# ------------------------------------------------------------- formatting
def _ok(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def count(x) -> str:  # 8213 -> 8.2K
    if not _ok(x):
        return NA
    return f"{x / 1000:.1f}K" if x >= 1000 else f"{x:.0f}"


def ms(x) -> str:
    if not _ok(x):
        return NA
    return f"{x:.1f}" if x >= 10 else f"{x:.2f}"


def pct(x, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f}%" if _ok(x) else NA


def rec(x) -> str:
    return f"{x:.3f}" if _ok(x) else NA


def gain(a, b) -> str:
    return f"{a / b:.2f}x" if _ok(a) and _ok(b) and b > 0 else NA


def size(nbytes) -> str:
    if not _ok(nbytes):
        return NA
    if nbytes >= 2 ** 20:
        return f"{nbytes / 2 ** 20:.1f} MB"
    return f"{nbytes / 2 ** 10:.0f} KB"


def get(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def load(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def table(title: str, header: list[str], rows: list[list[str]], notes: list[str] = ()):
    if not rows:
        return
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]

    def line(cells):
        first = str(cells[0]).ljust(widths[0])
        rest = (str(c).rjust(w) for c, w in zip(cells[1:], widths[1:]))
        return "  ".join([first, *rest]).rstrip()

    print(f"\n{title}")
    print(line(header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(line(r))
    for n in notes:
        print(f"  {n}")


# ----------------------------------------------------------------- tables
def table_ii(R: Path):
    rows, notes = [], []
    d = load(R / "cost" / "disk_video.json")
    if d:
        c, t = d["cold"], d["trace"]
        rows += [
            ["Disk: Ego4D -> LAION-10M", "", "", ""],
            ["  4 KB node-page fetches", count(c["reads_mean"]), count(t["reads_mean"]),
             gain(c["reads_mean"], t["reads_mean"])],
            ["  full-vector distances", count(c["dist_mean"]), count(t["dist_mean"]),
             gain(c["dist_mean"], t["dist_mean"])],
            ["  p50 / p95 latency (ms)", f"{ms(c['p50_ms'])} / {ms(c['p95_ms'])}",
             f"{ms(t['p50_ms'])} / {ms(t['p95_ms'])}",
             f"{gain(c['p50_ms'], t['p50_ms'])} / {gain(c['p95_ms'], t['p95_ms'])}"],
            ["  p99 latency (ms)", ms(c["p99_ms"]), ms(t["p99_ms"]),
             gain(c["p99_ms"], t["p99_ms"])],
            ["  CPU time / I/O wait at p50 (ms)",
             f"{ms(c['cpu_p50_ms'])} / {ms(c['io_p50_ms'])}",
             f"{ms(t['cpu_p50_ms'])} / {ms(t['io_p50_ms'])}", ""],
            ["  recall@10", rec(c.get("recall_mean")), rec(t.get("recall_mean")), ""],
        ]
        ci = d.get("ci", {})
        notes.append(f"disk fetch gain over {ci.get('groups', NA)} session groups: "
                     f"{ci.get('gain_mean', float('nan')):.2f} "
                     f"+/- {ci.get('gain_ci95', float('nan')):.2f} (95% CI)")
        key = "deadline_miss_33.3ms"
        notes.append(f"disk 33 ms budget misses (Sec. V-H): cold {pct(c.get(key), 1)}, "
                     f"TRACE {pct(t.get(key), 1)}")
    for tier, name in (("mem_video", "Memory: Ego4D -> LAION-1M (512-d)"),
                       ("mem_dialog", "Memory: dialogue turns -> Wikipedia-1M (384-d)")):
        d = load(R / "cost" / f"{tier}.json")
        if not d:
            continue
        c, t = d["cold"], d["trace"]
        rows += [
            [name, "", "", ""],
            ["  distance computations", count(c["dist_mean"]), count(t["dist_mean"]),
             gain(c["dist_mean"], t["dist_mean"])],
            ["  p50 / p95 latency (ms)", f"{ms(c['p50_ms'])} / {ms(c['p95_ms'])}",
             f"{ms(t['p50_ms'])} / {ms(t['p95_ms'])}",
             f"{gain(c['p50_ms'], t['p50_ms'])} / {gain(c['p95_ms'], t['p95_ms'])}"],
            ["  p99 latency (ms)", ms(c["p99_ms"]), ms(t["p99_ms"]),
             gain(c["p99_ms"], t["p99_ms"])],
            ["  recall@10", rec(c.get("recall_mean")), rec(t.get("recall_mean")), ""],
        ]
        ci = d.get("ci", {})
        notes.append(f"{tier} distance gain over {ci.get('groups', NA)} session groups: "
                     f"{ci.get('gain_mean', float('nan')):.2f} "
                     f"+/- {ci.get('gain_ci95', float('nan')):.2f} (95% CI)")
    table("Table II. Per-query cost at matched mean recall@10 (TRACE-fast, one thread)",
          ["Metric (per query)", "Cold", "TRACE", "Gain"], rows, notes)


def table_iii(R: Path):
    bv = load(R / "baselines" / "mem_video.json") or {}
    br = load(R / "baselines" / "mem_dialog.json") or {}
    bd = load(R / "baselines" / "disk_video.json") or {}
    rows = []
    for m, name in METHOD_NAMES.items():
        if m not in bv and m not in br and m not in bd:
            continue
        rows.append([name,
                     count(get(bv, m, "dist_mean")), ms(get(bv, m, "p50_ms")),
                     count(get(br, m, "dist_mean")), ms(get(br, m, "p50_ms")),
                     count(get(bd, m, "reads_mean")), ms(get(bd, m, "p99_ms")),
                     ", ".join(rec(get(b, m, "recall_mean")) for b in (bv, br, bd))])
    table("Table III. Baselines at matched mean recall@10",
          ["Method", "Video Dist.", "p50", "Dialogue Dist.", "p50", "Disk Reads", "p99",
           "recall (V, D, disk)"], rows)


def table_iv(R: Path):
    am = load(R / "attribution" / "mem_video.json") or {}
    ad = load(R / "attribution" / "disk_video.json") or {}
    rows = []
    for m, name in ATTRIBUTION_ROWS:
        if m not in am and m not in ad:
            continue
        rows.append([name, count(get(am, m, "dist_mean")), ms(get(am, m, "p50_ms")),
                     f"{rec(get(am, m, 'recall_mean'))} / {rec(get(am, m, 'recall_worst100'))}",
                     count(get(ad, m, "reads_mean")), ms(get(ad, m, "p50_ms")),
                     f"{rec(get(ad, m, 'recall_mean'))} / {rec(get(ad, m, 'recall_worst100'))}"])
    notes = []
    for tier, a in (("memory", am), ("disk", ad)):
        ss = a.get("same_seeds")
        if ss and ss.get("n_seeded"):
            notes.append(
                f"{tier}, seeds held fixed on {ss['n_seeded']} seeded-path queries: distances "
                f"{count(ss['dist_prune'])} with pruning vs {count(ss['dist_noprune'])} "
                f"without, p50 {ms(ss['p50_ms_prune'])} vs {ms(ss['p50_ms_noprune'])} ms, "
                f"recall {rec(ss.get('recall_mean'))}, identical result lists: "
                f"{ss['identical_results']} ({ss.get('prune_equiv_fail', NA)} mismatches)")
    table("Table IV. Cumulative attribution on video (recall: mean / worst 100-query window)",
          ["Configuration", "Mem Dist.", "p50", "Recall@10", "Disk Reads", "p50",
           "Recall@10"], rows, notes)
    # Eq. (6): per-path cost decomposition of TRACE's distance work (Sec. V-D)
    header = False
    for tier in ("mem_video", "mem_dialog"):
        c = load(R / "cost" / f"{tier}.json")
        if not c:
            continue
        t, cold = c["trace"], c["cold"]["dist_mean"]
        res, seed, fb = t["frac_resolve"], t["frac_seeded"], t["frac_cold"]
        bound_res = (M_ANCHORS + K_RESULT + S_MAX) / cold  # (m + k + s_max) / V_cold
        c_seed = (t["dist_mean"] / cold - res * bound_res - fb) / max(seed, 1e-9)
        if not header:
            print("\nSec. V-D, Eq. (6). Path decomposition of TRACE's distance work")
            header = True
        print(f"  Eq. (6), {tier}: path frequencies (resolve, seeded, fallback) = "
              f"({pct(res)}, {pct(seed)}, {pct(fb)}); estimated seeded-path cost "
              f"{c_seed:.2f} of cold (c_res at its bound, c_fb = 1)")


def table_v(R: Path):
    rows = []
    for label, name in STORAGE_ROWS:
        d = load(R / "storage" / f"disk_video_{label}.json")
        if not d:
            continue
        c, t = d["cold"], d["trace"]
        key = "reads_mean" if d.get("direct_io") else "device_reads_mean"
        rows.append([name, f"{count(c.get(key))} / {count(t.get(key))}",
                     f"{ms(c['p50_ms'])} / {ms(t['p50_ms'])}",
                     f"{ms(c['p99_ms'])} / {ms(t['p99_ms'])}"])
    table("Table V. Disk tier (Ego4D -> LAION-10M) under different caching configurations "
          "(cold / TRACE)", ["Configuration", "Device reads", "p50", "p99"], rows)


def table_vi(R: Path):
    d = load(R / "locality" / "mem_video.json")
    if not d:
        return
    rows = []
    for key, name in LOCALITY_ROWS:
        if key not in d:
            continue
        c, t = d[key]["cold"], d[key]["trace"]
        rows.append([name, pct(t["frac_resolve"]), pct(t["frac_cold"]), count(t["dist_mean"]),
                     ms(t["p50_ms"]), rec(t.get("recall_mean")), gain(c["p50_ms"], t["p50_ms"])])
    table("Table VI. Locality sweep on the controlled stream (memory, 512-d)",
          ["rho bin", "Resolve", "Cold", "Dist.", "p50 (ms)", "Recall@10", "p50 gain"], rows,
          ["Resolve and Cold are path frequencies; the seeded path takes the rest."])


def table_vii(R: Path):
    d = load(R / "dynamics" / "mem_video.json")
    if not d:
        return
    rows = []
    for key, name, recency, hybrid in DYNAMICS_ROWS:
        s = d.get(key, {})
        if hybrid not in s:
            continue
        h = s[hybrid]
        rows.append([name, count(get(s, "cold", "dist_mean")), count(get(s, recency, "dist_mean")),
                     count(h["dist_mean"]), pct(h["frac_resolve"]), rec(h.get("recall_mean"))])
    table("Table VII. Stream dynamics on the video tier (memory)",
          ["Stream", "Cold", "Recency", "Hybrid", "Resolve", "Recall@10"], rows,
          ["Distance computations per retention policy; resolve rate and recall for hybrid."])


def table_viii(R: Path):
    d = load(R / "sensitivity" / "mem_video.json")
    if not d:
        return
    rows = []
    for m, name in SENSITIVITY_ROWS:
        if m not in d:
            continue
        s = d[m]
        if m == "cold":
            rows.append([name, count(s["dist_mean"]), NA, NA, ms(s["p50_ms"])])
        else:
            rows.append([name, count(s["dist_mean"]), pct(s["frac_resolve"]),
                         pct(s.get("coverage_miss_frac"), 1), ms(s["p50_ms"])])
    table("Table VIII. Sensitivity (video, memory, default operating point, not recall-matched)",
          ["Configuration", "Dist.", "Resolved", "Miss", "p50 (ms)"], rows,
          ["Miss: share of resolved answers whose retained set lacked a true neighbor."])


def guarantees(R: Path):
    rows = []
    for tier in ("mem_video", "mem_dialog", "disk_video"):
        d = load(R / "guarantees" / f"{tier}.json")
        if not d:
            continue
        for m in ("trace", "trace_safe"):
            s = d.get(m)
            if not s:
                continue
            chain = s.get("resolved_recall_by_chain", {})
            resolved = s["n_queries"] * s["frac_resolve"]
            rows.append([f"{tier} {m}", f"{resolved:.0f}", f"{s['audit_checked']:.0f}",
                         f"{s['audit_violations']:.0f}", f"{s['resolve_exact_fail']:.0f}",
                         pct(s.get("recall_full_frac")), rec(s.get("recall_p5")),
                         rec(chain.get("1")), rec(chain.get(">8"))])
        c = d.get("cold")
        if c:
            rows.append([f"{tier} cold", NA, NA, NA, NA, pct(c.get("recall_full_frac")),
                         rec(c.get("recall_p5")), NA, NA])
    table("Sec. V-G. Guarantees in practice",
          ["Tier / method", "Resolved", "Bounds audited", "Violations", "Theorem 1 fails",
           "All 10 found", "Recall p5", "Resolved recall, chain 1", "chain >8"], rows,
          ["Bounds audited: lower bounds of retained nodes checked against an exhaustive",
           "float32 rescoring of R (Lemma 1); Violations: bounds above s(q, x).",
           "Theorem 1 fails: resolved answers that differ from the exact top-k of R."])


def table_ix(R: Path):
    d = load(R / "e2e" / "topiocqa.json")
    if not d:
        return
    rows = []
    for m, name in (("exact", "exact search"), ("cold", "cold search"), ("trace", "TRACE")):
        s = d.get(m, {})
        rows.append([name, f"{s.get('f1', float('nan')):.1f}",
                     rec(s.get("recall@5", 1.0 if m == "exact" else None)),
                     ms(s.get("retrieval_p99_ms")), ms(s.get("ttft_p50_ms"))])
    diff = d.get("f1_diff_trace_minus_cold", {})
    lo, hi = diff.get("ci95", [float("nan"), float("nan")])
    table("Table IX. End-to-end conversational RAG on TopiOCQA (top-5 context)",
          ["Retrieval", "Answer F1", "Recall@5", "Retrieval p99 (ms)", "TTFT p50 (ms)"], rows,
          [f"F1 difference TRACE - cold: {diff.get('mean', float('nan')):.2f}, "
           f"95% conversation-bootstrap interval [{lo:.2f}, {hi:.2f}]"])


def table_x(R: Path):
    sc = load(R / "scale" / "mem_video.json")
    if sc:
        sessions = sorted({int(k.split("_S")[1]) for k in sc if k.startswith("trace_S")})
        rows = [
            ["Cold QPS", *(count(get(sc, f"cold_S{s}", "qps")) for s in sessions)],
            ["TRACE QPS", *(count(get(sc, f"trace_S{s}", "qps")) for s in sessions)],
            ["Throughput gain", *(gain(get(sc, f"trace_S{s}", "qps"), get(sc, f"cold_S{s}", "qps"))
                                  for s in sessions)],
            ["Residue memory", *(size(get(sc, f"trace_S{s}", "state_bytes")) for s in sessions)],
        ]
        table(f"Table X (top). Concurrent sessions on {sc.get('threads', NA)} threads "
              "(video, memory)", ["Sessions", *(str(s) for s in sessions)], rows)
    rows = []
    for tier, name in WIDTH_ROWS:
        d = load(R / "width" / f"{tier}.json")
        if not d:
            continue
        rows.append([name, gain(d["cold"]["dist_mean"], d["trace"]["dist_mean"]),
                     pct(d["trace"]["frac_resolve"]), size(d.get("ring_bytes_per_session"))])
    table("Table X (bottom). Embedding width on the dialogue corpus (each re-tuned)",
          ["Embedding width", "Dist. gain", "Resolved", "Ring payload"], rows)


def fidelity(R: Path):
    d = load(R / "fidelity" / "fid_qvcache_deep.json")
    if d:
        s, rep = d["summary"], d.get("reported", {})
        rows = [
            ["hit ratio", pct(s["hit_ratio"]), NA],
            ["hit-ratio drop at a working-set shift",
             f"{s['hit_ratio_drop_at_shift_pct']:.0f}%",
             f"{rep.get('hit_ratio_drop_at_shift_pct', float('nan')):.0f}%"],
            ["recall@10 loss (relative)", f"{s['recall_drop_pct']:.1f}%",
             "{:.0f}-{:.0f}%".format(*rep.get("recall_drop_pct", [math.nan, math.nan]))],
            ["hit latency p50 (ms)", ms(s["hit_latency_p50_ms"]),
             "{}-{}".format(*rep.get("hit_latency_ms", [NA, NA]))],
        ]
        table("Sec. V-A. QVCache fidelity (DEEP-10M, its generated workload)",
              ["Metric", "Ours", "Reported"], rows, [f"source: {rep.get('source', NA)}"])
    d = load(R / "fidelity" / "fid_catapult_pubmed.json")
    if d:
        rep = d.get("reported", {})
        rows = []
        for k, v in d.items():
            if not isinstance(v, dict) or "summary" not in v:
                continue
            s, r = v["summary"], rep.get(k, {})
            rows.append([f"k={k}", f"{s['hops_reduction_pct']:.1f}%",
                         _fmt_reported(r, "hops_reduction_pct"), f"{s['usage_pct']:.1f}%",
                         _fmt_reported(r, "usage_pct"),
                         f"{rec(s['recall_cold'])} / {rec(s['recall_catapult'])}"])
        table("Sec. V-A. CatapultDB fidelity (PubMed MedCPT-1M, uniform queries)",
              ["k", "Node-visit reduction", "Reported", "Catapult usage", "Reported",
               "Recall (cold / CatapultDB)"], rows, [f"source: {rep.get('source', NA)}"])


def _fmt_reported(r: dict, key: str) -> str:
    v = r.get(key) if isinstance(r, dict) else None
    return f"{v:.1f}%" if _ok(v) else NA


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results", help="results directory")
    R = Path(ap.parse_args().results)
    for fn in (table_ii, table_iii, table_iv, table_v, table_vi, table_vii, table_viii,
               guarantees, table_ix, table_x, fidelity):
        fn(R)


if __name__ == "__main__":
    main()
