#!/usr/bin/env python3
"""Builds a DiskANN disk index (Vamana, full-precision vectors on SSD, PQ codes
in RAM) with diskannpy and checks the layout TRACE's reader expects."""
import argparse
import struct
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401
from trace_harness.io import read_bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help=".fbin (float32, unit-norm)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="ann")
    ap.add_argument("--R", type=int, default=64, help="graph degree")
    ap.add_argument("--L", type=int, default=128, help="build complexity")
    ap.add_argument("--pq-bytes", type=int, default=64, help="PQ code bytes per vector (RAM)")
    ap.add_argument("--build-ram-gb", type=float, default=64.0)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    import os

    import diskannpy

    n, d = read_bin(a.corpus).shape
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # DiskANN derives the number of PQ chunks as floor(budget_bytes / n);
    # half a byte of slack avoids flooring 64.0 to 63.
    budget_gb = (a.pq_bytes + 0.5) * n / 2**30
    diskannpy.build_disk_index(
        data=a.corpus, distance_metric="l2", index_directory=str(out),
        complexity=a.L, graph_degree=a.R, search_memory_maximum=budget_gb,
        build_memory_maximum=a.build_ram_gb, num_threads=a.threads or os.cpu_count(),
        pq_disk_bytes=0, vector_dtype=np.float32, index_prefix=a.prefix)
    pre = out / a.prefix
    with open(f"{pre}_pq_compressed.bin", "rb") as f:
        npts, nch = struct.unpack("<ii", f.read(8))
    with open(f"{pre}_disk.index", "rb") as f:
        nr, nc = struct.unpack("<ii", f.read(8))
        meta = struct.unpack(f"<{nr}Q", f.read(8 * nr))
    print(f"points={npts} pq_chunks={nch} meta={meta}")
    assert npts == n and nch == a.pq_bytes, "unexpected PQ size"
    assert meta[0] == n and meta[1] == d and meta[5] == 0
    print("max_node_len", meta[3], "nodes/sector", meta[4])


if __name__ == "__main__":
    main()
