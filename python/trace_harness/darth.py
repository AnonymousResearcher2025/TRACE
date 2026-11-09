"""DARTH-style early termination: training data, GBDT training, export.

At every check (every ``interval`` full-vector evaluations) the executor logs
the progress features (see ``darth_features`` in baselines.cpp) and the
current top-k. The label is the recall@k of that top-k against ground truth.
A GBDT regressor predicts recall; at inference the search stops once the
prediction reaches the declared target (DARTH: Chatzakis et al., PACMMOD 3(4),
2025)."""
from __future__ import annotations

import json

import numpy as np

import trace_harness as th


def collect(executor, Q: np.ndarray, sessions: np.ndarray, gt_ids: np.ndarray, *,
            k: int, L: int, interval: int, beam_width: int = 4):
    """Runs cold searches with a logging policy; returns (X, y)."""
    log = th.LogStop(interval)
    log.k = k
    cfg = th.TraceConfig()
    cfg.enabled = False
    cfg.k, cfg.L, cfg.beam_width = k, L, beam_width
    eng = th.TraceEngine(executor, cfg, None, log)
    th.replay(eng, Q, sessions, k=k, log=log)
    X = log.features()
    ids = log.topk()
    qi = log.query_index()
    labels = executor.labels(ids)
    y = np.empty(len(qi), dtype=np.float32)
    for r in range(len(qi)):
        got = labels[r][labels[r] >= 0]
        y[r] = np.intersect1d(got, gt_ids[qi[r], :k]).size / k
    return X, y


def train(X: np.ndarray, y: np.ndarray, path: str, *, n_estimators: int = 100,
          num_leaves: int = 31, learning_rate: float = 0.1, seed: int = 0) -> None:
    import lightgbm as lgb  # type: ignore

    model = lgb.LGBMRegressor(n_estimators=n_estimators, num_leaves=num_leaves,
                              learning_rate=learning_rate, random_state=seed,
                              verbose=-1)
    model.fit(X, y)
    export_lightgbm(model.booster_, path)
    # The exported model must reproduce LightGBM's predictions.
    ref = model.booster_.predict(X[:5000])
    ours = predict_file(path, X[:5000])
    assert np.allclose(ref, ours, atol=1e-4), float(np.abs(ref - ours).max())


def export_lightgbm(booster, path: str) -> None:
    dump = booster.dump_model()
    trees = []
    for t in dump["tree_info"]:
        tree = ([], [], [], [], [])  # feat, thr, left, right, value (pre-order)
        _flatten(t["tree_structure"], tree)
        trees.append(tree)
    _write(path, 0.0, trees)


def _flatten(node: dict, tree: tuple) -> int:
    """Appends ``node`` and its subtree to ``tree`` in pre-order; returns its index."""
    feat, thr, left, right, val = tree
    i = len(feat)
    feat.append(-1)
    thr.append(0.0)
    left.append(-1)
    right.append(-1)
    val.append(0.0)
    if "leaf_value" in node and "split_feature" not in node:
        val[i] = float(node["leaf_value"])
        return i
    assert node.get("decision_type", "<=") == "<=", node.get("decision_type")
    feat[i] = int(node["split_feature"])
    # Largest float32 <= threshold: for float32 features x,
    # x <= t32 holds exactly when x <= threshold.
    t = float(node["threshold"])
    t32 = np.float32(t)
    if float(t32) > t:
        t32 = np.nextafter(t32, np.float32(-np.inf))
    thr[i] = float(t32)
    left[i] = _flatten(node["left_child"], tree)
    right[i] = _flatten(node["right_child"], tree)
    return i


def write_constant_model(path: str, value: float) -> None:
    _write(path, float(value), [([-1], [0.0], [-1], [-1], [0.0])])


def _write(path, base, trees):
    with open(path, "w") as f:
        f.write("GBDT 1\n")
        f.write(f"base {base!r}\n")
        f.write(f"trees {len(trees)}\n")
        for feat, thr, left, right, val in trees:
            f.write(f"tree {len(feat)}\n")
            for a, b, c, d, e in zip(feat, thr, left, right, val):
                f.write(f"{a} {b!r} {c} {d} {e!r}\n")


def predict_file(path: str, X: np.ndarray) -> np.ndarray:
    """Python reference evaluator of the exported format (float32 compares,
    as in the C++ evaluator)."""
    with open(path) as f:
        tok = f.read().split()
    p = 0

    def nxt():
        nonlocal p
        p += 1
        return tok[p - 1]

    assert nxt() == "GBDT" and nxt() == "1"
    assert nxt() == "base"
    base = np.float32(float(nxt()))
    assert nxt() == "trees"
    trees = []
    for _ in range(int(nxt())):
        assert nxt() == "tree"
        nn = int(nxt())
        rows = [(int(nxt()), np.float32(float(nxt())), int(nxt()), int(nxt()),
                 np.float32(float(nxt()))) for _ in range(nn)]
        trees.append(rows)
    X = np.asarray(X, dtype=np.float32)
    out = np.empty(len(X), dtype=np.float64)
    for r, x in enumerate(X):
        s = np.float32(base)
        for rows in trees:
            n = 0
            while rows[n][0] >= 0:
                f_, t_, l_, r_, _ = rows[n]
                n = l_ if x[f_] <= t_ else r_
            s = np.float32(s + rows[n][4])
        out[r] = s
    return out


def save_meta(path: str, **kw) -> None:
    with open(path, "w") as f:
        json.dump(kw, f, indent=2)


__all__ = ["collect", "train", "export_lightgbm", "write_constant_model",
           "predict_file", "save_meta"]
