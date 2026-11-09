#!/usr/bin/env python3
"""Runs one experiment of the paper.

Examples
  python scripts/run.py -c configs/aws.yaml cost --tiers mem_video mem_dialog disk_video
  python scripts/run.py -c configs/aws.yaml baselines --tiers mem_video
  sudo systemd-run --scope -p MemoryMax=2G -p MemorySwapMax=0 \\
      python scripts/run.py -c configs/aws.yaml storage --tiers disk_video --label buffered_2g
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from trace_harness.experiments import Runner  # noqa: E402
from trace_harness.runner import load_config  # noqa: E402

EXPS = ["cost", "baselines", "attribution", "storage", "locality", "dynamics",
        "sensitivity", "guarantees", "scale", "width", "fidelity"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("experiment", choices=EXPS)
    ap.add_argument("--tiers", nargs="+", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N replay queries (quick partial runs)")
    ap.add_argument("--label", default=None, help="storage: configuration label")
    ap.add_argument("--methods", nargs="*", default=None, help="baselines: subset")
    ap.add_argument("--threads", type=int, default=16, help="scale: worker threads")
    ap.add_argument("--sessions", type=int, nargs="+", default=[1, 16, 256],
                    help="scale: concurrent session counts")
    a = ap.parse_args()
    r = Runner(load_config(a.config), a.results, reps=a.reps, limit=a.limit)
    if a.experiment == "storage":
        assert a.label, "--label is required for storage"
        r.storage(a.tiers, a.label)
    elif a.experiment == "baselines":
        r.baselines(a.tiers, a.methods)
    elif a.experiment == "scale":
        r.scale(a.tiers, sessions=tuple(a.sessions), threads=a.threads)
    else:
        getattr(r, a.experiment)(a.tiers)


if __name__ == "__main__":
    main()
