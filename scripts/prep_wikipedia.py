#!/usr/bin/env python3
"""English Wikipedia passages: 128-token windows with 32-token overlap (MiniLM
tokenizer), from articles visited in a fixed-seed random order, embedded with
--model and unit-normalized.

Pass 1 (default): chunk + embed; writes passages.tsv (id, title, text) and
corpus_<N>.fbin, plus a fixed-seed corpus_<Nsmall>.fbin subset.
Pass 2 (--passages FILE): re-embed an existing passages.tsv with another
model (width study: all-mpnet-base-v2 = 768-d, openai:text-embedding-3-small
= 1536-d)."""
import argparse
import csv
from pathlib import Path

import numpy as np

from _common import fbin_writer
from _encoders import Encoder

csv.field_size_limit(1 << 30)


def chunks(tok, text, size, overlap):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    step = size - overlap
    for s in range(0, max(len(ids) - overlap, 1), step):
        w = ids[s:s + size]
        if len(w) < 16:  # skip tiny tails
            break
        yield tok.decode(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--n-small", type=int, default=1_000_000)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--tokenizer", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--passages", default=None, help="re-embed this passages.tsv")
    ap.add_argument("--tag", default="", help="suffix for output corpus files")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    enc = Encoder(a.model, a.device, a.batch)
    d = enc.dim()
    rng = np.random.default_rng(a.seed)
    corpus = fbin_writer(out / f"corpus_{a.n}{a.tag}.fbin", a.n, d)
    pos, buf = 0, []

    def flush():
        nonlocal pos, buf
        if buf:
            e = enc.encode(buf)
            corpus[pos:pos + len(e)] = e
            pos += len(e)
            buf = []
            print(f"{pos:,}/{a.n:,}", flush=True)

    if a.passages:
        with open(a.passages, newline="") as f:
            for row in csv.reader(f, delimiter="\t"):
                if pos + len(buf) >= a.n:
                    break
                buf.append(row[2])
                if len(buf) == 8192:
                    flush()
        flush()
    else:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
        order = rng.permutation(len(ds))
        with open(out / "passages.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            for ai in order:
                art = ds[int(ai)]
                for text in chunks(tok, art["text"], a.chunk, a.overlap):
                    if pos + len(buf) >= a.n:
                        break
                    flat = text.replace("\t", " ").replace("\n", " ")
                    w.writerow([pos + len(buf), art["title"], flat])
                    buf.append(text)
                    if len(buf) == 8192:
                        flush()
                if pos + len(buf) >= a.n:
                    break
            flush()
    assert pos == a.n, f"only {pos} passages produced"
    corpus.flush()
    sub_path = out / f"subset_{a.n_small}_rows.npy"
    if sub_path.exists():
        sub = np.load(sub_path)
    else:
        sub = np.sort(rng.choice(a.n, size=a.n_small, replace=False))
        np.save(sub_path, sub)
    small = fbin_writer(out / f"corpus_{a.n_small}{a.tag}.fbin", a.n_small, d)
    for s in range(0, a.n_small, 1_000_000):
        small[s:s + 1_000_000] = corpus[sub[s:s + 1_000_000]]
    small.flush()
    print("done")


if __name__ == "__main__":
    main()
