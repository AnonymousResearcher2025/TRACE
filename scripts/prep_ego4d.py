#!/usr/bin/env python3
"""Ego4D frame stream: frames sampled at --fps from each clip, embedded with
OpenAI CLIP ViT-B/32 (the model behind the LAION-400M embeddings), unit-norm.
Each clip is one session. Output: a workload directory (queries.npy,
sessions.npy, meta.json)."""
import argparse
import subprocess
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401
from trace_harness.io import l2_normalize, save_workload


def frames(path, fps):
    """Yields PIL images decoded by ffmpeg at the requested frame rate."""
    from PIL import Image

    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0",
                            str(path)], capture_output=True, text=True, check=True)
    w, h = map(int, probe.stdout.strip().split(",")[:2])
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield Image.frombytes("RGB", (w, h), buf)
    p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="directory of clips or a text file of paths")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--max-queries", type=int, default=57_000,
                    help="tuning prefix + warmup + 50K scored queries")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(a.model, pretrained=a.pretrained)
    model = model.to(a.device).eval()
    src = Path(a.videos)
    if src.is_dir():
        vids = sorted(src.glob("*.mp4"))
    else:
        with open(src) as f:
            vids = [Path(line.strip()) for line in f if line.strip()]
    Q, S, names = [], [], []
    total = 0
    for ci, v in enumerate(vids):
        embs, batch = [], []
        for img in frames(v, a.fps):
            batch.append(preprocess(img))
            if len(batch) == a.batch:
                with torch.no_grad():
                    embs.append(model.encode_image(torch.stack(batch).to(a.device)).float().cpu().numpy())
                batch = []
        if batch:
            with torch.no_grad():
                embs.append(model.encode_image(torch.stack(batch).to(a.device)).float().cpu().numpy())
        if not embs:
            continue
        e = l2_normalize(np.concatenate(embs))
        Q.append(e)
        S.append(np.full(len(e), ci, dtype=np.int64))
        names.append(v.name)
        total += len(e)
        print(f"{v.name}: {len(e)} frames ({total:,})", flush=True)
        if total >= a.max_queries:
            break
    Q = np.concatenate(Q)[: a.max_queries]
    S = np.concatenate(S)[: a.max_queries]
    save_workload(a.out, Q, S, {"source": "ego4d", "fps": a.fps, "model": a.model,
                                "pretrained": a.pretrained, "clips": names})


if __name__ == "__main__":
    main()
