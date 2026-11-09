#!/usr/bin/env python3
"""Corpora and query streams for the fidelity checks of the two
reimplemented baselines (trace_harness/fidelity.py).

  deep    DEEP-10M (first 10M of Deep1B, 96-d, L2) + QVCache's generated
          stream from the 10K public queries (84,000 queries).
          Get the inputs with big-ann-benchmarks:
            python create_dataset.py --dataset deep-10M
          -> data/deep1b/base.1B.fbin.crop_nb_10000000, query.public.10K.fbin
  pubmed  MedCPT embeddings of the first 1M PubMed documents (NCBI release)
          + CatapultDB's Uniform stream (150,000 queries in [-1, 1]^768).
            wget https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/embeds_chunk_0.npy
          (add embeds_chunk_1.npy if chunk 0 holds fewer than 1M rows)

Writes <out>/corpus_<n>.fbin and the workload directory <workload>
(queries.npy, sessions.npy, meta.json; deep also dispatch.npy and
base_index.npy). Then run ground_truth.py on the pair.
"""
import argparse
from pathlib import Path

import numpy as np

from _common import fbin_writer
from trace_harness.fidelity import QVCACHE_DEFAULTS, qvcache_stream, uniform_stream
from trace_harness.io import read_bin, save_workload

CHUNK = 1_000_000


def copy_rows(src, dst_path, n):
    d = src.shape[1]
    out = fbin_writer(dst_path, n, d)
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        out[a:b] = np.asarray(src[a:b], dtype=np.float32)
    out.flush()
    return read_bin(dst_path)


def deep(a):
    base = read_bin(a.base)
    assert base.shape[0] >= a.n, f"{a.base} has {base.shape[0]} rows < {a.n}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    X = copy_rows(base, out / f"corpus_{a.n}.fbin", a.n)
    Q = np.asarray(read_bin(a.queries), dtype=np.float32)
    p = dict(QVCACHE_DEFAULTS)
    Qs, disp, bidx = qvcache_stream(Q, X, seed=a.seed, **p)
    save_workload(a.workload, Qs, np.zeros(len(Qs), np.int64),
                  meta={"generator": "qvcache", **p, "seed": a.seed, "corpus": str(out),
                        "base_queries": a.queries})
    np.save(Path(a.workload) / "dispatch.npy", disp)
    np.save(Path(a.workload) / "base_index.npy", bidx)
    print(f"corpus {X.shape}, stream {Qs.shape}, dispatches {disp.max() + 1}")


def pubmed(a):
    parts, have = [], 0
    for f in a.embeds:
        e = np.load(f, mmap_mode="r")
        parts.append(e)
        have += e.shape[0]
        if have >= a.n:
            break
    assert have >= a.n, f"only {have} rows in {a.embeds}"
    d = parts[0].shape[1]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    X = fbin_writer(out / f"corpus_{a.n}.fbin", a.n, d)
    pos = 0
    for e in parts:
        take = min(e.shape[0], a.n - pos)
        for s in range(0, take, CHUNK):
            t = min(take, s + CHUNK)
            X[pos + s:pos + t] = np.asarray(e[s:t], dtype=np.float32)
        pos += take
    X.flush()
    U = uniform_stream(a.queries, d, seed=a.seed)
    save_workload(a.workload, U, np.zeros(len(U), np.int64),
                  meta={"generator": "uniform", "low": -1.0, "high": 1.0, "seed": a.seed,
                        "corpus": str(out)})
    print(f"corpus ({a.n}, {d}), stream {U.shape}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("deep")
    p.add_argument("--base", required=True, help="Deep1B base .fbin (>= n rows)")
    p.add_argument("--queries", required=True, help="query.public.10K.fbin")
    p.add_argument("--out", required=True)
    p.add_argument("--workload", required=True)
    p.add_argument("--n", type=int, default=10_000_000)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("pubmed")
    p.add_argument("--embeds", nargs="+", required=True, help="embeds_chunk_*.npy in order")
    p.add_argument("--out", required=True)
    p.add_argument("--workload", required=True)
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--queries", type=int, default=150_000)
    p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    deep(a) if a.cmd == "deep" else pubmed(a)


if __name__ == "__main__":
    main()
