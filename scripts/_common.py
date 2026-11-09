"""Shared helpers for scripts (adds python/ to sys.path)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def fbin_writer(path, n, d):
    """Creates an .fbin file and returns a writable float32 memmap view."""
    import numpy as np

    with open(path, "wb") as f:
        np.array([n, d], dtype=np.int32).tofile(f)
        f.truncate(8 + 4 * n * d)
    return np.memmap(path, dtype=np.float32, mode="r+", offset=8, shape=(n, d))
