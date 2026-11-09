"""File formats.

* ``.fbin`` / ``.ibin`` (DiskANN): int32 n, int32 d, then n*d values.
* ``.npy``: numpy arrays (memory-mapped when large).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def write_bin(path: str | os.PathLike, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr)
    assert arr.ndim == 2
    with open(path, "wb") as f:
        np.array(arr.shape, dtype=np.int32).tofile(f)
        arr.tofile(f)


def read_bin(path: str | os.PathLike, dtype=np.float32, mmap: bool = True) -> np.ndarray:
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype=np.int32, count=2)
    if mmap:
        return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(int(n), int(d)))
    with open(path, "rb") as f:
        f.seek(8)
        return np.fromfile(f, dtype=dtype, count=int(n) * int(d)).reshape(int(n), int(d))


def l2_normalize(x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Row-normalizes in float32 (norms computed in float64)."""
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x.astype(np.float64), axis=1, keepdims=True)
    n[n == 0] = 1.0
    y = (x / n).astype(np.float32)
    if out is not None:
        out[:] = y
        return out
    return y


def save_json(path: str | os.PathLike, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_json(path: str | os.PathLike):
    with open(path) as f:
        return json.load(f)


class Workload:
    """A query stream: queries (n x d float32), session ids (n int64), and
    ground truth (n x K int64 labels, n x K float32 distances)."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.Q = np.load(self.root / "queries.npy", mmap_mode="r")
        self.sessions = np.load(self.root / "sessions.npy")
        gt = self.root / "gt_ids.npy"
        self.gt_ids = np.load(gt) if gt.exists() else None
        gd = self.root / "gt_dist.npy"
        self.gt_dist = np.load(gd) if gd.exists() else None
        meta = self.root / "meta.json"
        self.meta = load_json(meta) if meta.exists() else {}

    def __len__(self) -> int:
        return int(self.Q.shape[0])

    def subset(self, idx: np.ndarray):
        sub = object.__new__(Workload)
        sub.root = self.root
        sub.Q = np.ascontiguousarray(self.Q[idx])
        sub.sessions = self.sessions[idx]
        sub.gt_ids = None if self.gt_ids is None else self.gt_ids[idx]
        sub.gt_dist = None if self.gt_dist is None else self.gt_dist[idx]
        sub.meta = self.meta
        return sub


def save_workload(root: str | os.PathLike, Q: np.ndarray, sessions: np.ndarray,
                  meta: dict | None = None) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "queries.npy", np.ascontiguousarray(Q, dtype=np.float32))
    np.save(root / "sessions.npy", np.asarray(sessions, dtype=np.int64))
    save_json(root / "meta.json", meta or {})
