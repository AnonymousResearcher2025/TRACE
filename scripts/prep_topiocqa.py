#!/usr/bin/env python3
"""TopiOCQA conversations as a dialogue query stream (one session per
conversation) and the end-to-end RAG evaluation set.

Query text = the last --history previous turns (question and answer) followed
by the current question. Writes a workload directory plus turns.jsonl with
the question, gold answers, and the row of each turn in the workload."""
import argparse
import json
from pathlib import Path

import numpy as np

from _encoders import Encoder
from trace_harness.io import save_workload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--splits", nargs="+", default=["train", "validation"])
    ap.add_argument("--history", type=int, default=2, help="previous turns in the query")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from datasets import load_dataset

    ds = load_dataset("McGill-NLP/TopiOCQA")
    enc = Encoder(a.model, a.device)
    texts, sess, turns = [], [], []
    conv_ids = {}
    for sp in a.splits:
        for row in ds[sp]:
            try:
                cid = (sp, int(row["Conversation_no"]))
                q = row["Question"]
                ctx = list(row["Context"])  # alternating previous questions and answers
                ans = [row["Answer"]] + [x for x in (row.get("Additional_answers") or [])
                                         if isinstance(x, str)]
            except KeyError as e:
                raise SystemExit(f"unexpected TopiOCQA schema, missing {e}") from e
            sid = conv_ids.setdefault(cid, len(conv_ids))
            tail = ctx[-2 * a.history:] if a.history > 0 else []
            texts.append(" ".join(tail + [q]))
            sess.append(sid)
            turns.append({"row": len(texts) - 1, "split": sp, "conversation": cid[1],
                          "turn": int(row["Turn_no"]), "question": q, "answers": ans})
    # one contiguous block per conversation, turns in order
    order = sorted(range(len(texts)), key=lambda i: (sess[i], turns[i]["turn"]))
    texts = [texts[i] for i in order]
    sess = [sess[i] for i in order]
    turns = [turns[i] for i in order]
    for r, t in enumerate(turns):
        t["row"] = r
    Q = enc.encode(texts)
    out = Path(a.out)
    save_workload(out, Q, np.asarray(sess), {"source": "topiocqa", "model": a.model,
                                             "history": a.history, "splits": a.splits})
    with open(out / "turns.jsonl", "w") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")
    print(f"{len(texts)} turns, {len(conv_ids)} conversations")


if __name__ == "__main__":
    main()
