#!/usr/bin/env python3
"""Fixed-seed sample of LAION-400M CLIP ViT-B/32 image embeddings.

Input: the released img_emb_*.npy files (float16, 512-d). Output:
  corpus_<N>.fbin       float32, unit-norm, N x 512 (DiskANN .fbin layout)
  corpus_<Nsmall>.fbin  fixed-seed subset of the large sample
  provenance.npz        (file index, row) of every sampled vector
"""
import argparse
import re
from pathlib import Path

import numpy as np

from _common import fbin_writer
from trace_harness.io import l2_normalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--n-small", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    files = sorted(Path(a.emb_dir).glob("img_emb_*.npy"),
                   key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    assert files, "no img_emb_*.npy files found"
    sizes = np.array([np.load(f, mmap_mode="r").shape[0] for f in files], dtype=np.int64)
    d = np.load(files[0], mmap_mode="r").shape[1]
    print(f"{len(files)} files, {sizes.sum():,} vectors, dim {d}")
    rng = np.random.default_rng(a.seed)
    counts = rng.multivariate_hypergeometric(sizes, a.n)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    big = fbin_writer(out / f"corpus_{a.n}.fbin", a.n, d)
    prov_f, prov_r = [], []
    pos = 0
    for fi, (f, c) in enumerate(zip(files, counts)):
        if c == 0:
            continue
        arr = np.load(f, mmap_mode="r")
        rows = np.sort(rng.choice(arr.shape[0], size=int(c), replace=False))
        big[pos:pos + c] = l2_normalize(np.asarray(arr[rows], dtype=np.float32))
        prov_f.append(np.full(c, fi, dtype=np.int32))
        prov_r.append(rows.astype(np.int64))
        pos += int(c)
        print(f"{f.name}: {c} sampled ({pos:,}/{a.n:,})", flush=True)
    assert pos == a.n
    big.flush()
    np.savez(out / "provenance.npz", file=np.concatenate(prov_f), row=np.concatenate(prov_r),
             files=np.array([f.name for f in files]))
    sub = np.sort(rng.choice(a.n, size=a.n_small, replace=False))
    small = fbin_writer(out / f"corpus_{a.n_small}.fbin", a.n_small, d)
    for s in range(0, a.n_small, 1_000_000):
        small[s:s + 1_000_000] = big[sub[s:s + 1_000_000]]
    small.flush()
    np.save(out / f"subset_{a.n_small}_rows.npy", sub)
    print("done")


if __name__ == "__main__":
    main()
