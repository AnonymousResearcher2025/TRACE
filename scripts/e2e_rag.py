#!/usr/bin/env python3
"""End-to-end conversational RAG on TopiOCQA (Table IX).

For each validation turn: retrieve the top-5 passages with cold search or
TRACE (same tier, tuned L), build a prompt, and generate an answer with
Llama-3.1-8B-Instruct (vLLM, greedy). Reports token F1 against the gold
answers, recall@5, retrieval p99, and time to first token (TTFT) measured as
retrieval latency + prefill time of the prompt (one-token generation).

L for cold search and TRACE is read from results/tuning/<tier>/ and tuned on
the tier's workload first if absent (same procedure as scripts/run.py)."""
import argparse
import collections
import csv
import json
import re
import string
import sys
import time
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401
import trace_harness as th
from trace_harness.gt import recall_at_k
from trace_harness.io import Workload, save_json
from trace_harness.methods import build
from trace_harness.runner import Tier, context, load_config, split, tuned

csv.field_size_limit(sys.maxsize)


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, golds):
    best = 0.0
    p = normalize(pred).split()
    for g in golds:
        t = normalize(g).split()
        common = collections.Counter(p) & collections.Counter(t)
        same = sum(common.values())
        if same == 0:
            continue
        prec, rec = same / len(p), same / len(t)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def load_passages(path, wanted):
    wanted = {int(x) for x in wanted}
    out = {}
    with open(path, newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            i = int(row[0])
            if i in wanted:
                out[i] = (row[1], row[2])
    return out


def prompt(history, question, passages):
    ctx = "\n\n".join(f"[{i + 1}] {t}: {p}" for i, (t, p) in enumerate(passages))
    hist = "\n".join(history[-4:])
    return ("Answer the question using the passages. Reply with a short answer.\n\n"
            f"Passages:\n{ctx}\n\nConversation so far:\n{hist}\n\nQuestion: {question}\nAnswer:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--tier", required=True, help="disk tier over the Wikipedia corpus")
    ap.add_argument("--workload", required=True,
                    help="TopiOCQA workload dir (with turns.jsonl and ground truth)")
    ap.add_argument("--passages", required=True, help="passages.tsv of the corpus")
    ap.add_argument("--results", default="results")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--k-ctx", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=10000, help="bootstrap resamples")
    ap.add_argument("--ttft-turns", type=int, default=500,
                    help="turns whose prompt prefill is timed one request at a time")
    a = ap.parse_args()
    cfg = load_config(a.config)
    tier = Tier.from_dict(a.tier, cfg["tiers"][a.tier])
    ctx = context(tier, k=10)
    wl = Workload(a.workload)
    with open(Path(a.workload) / "turns.jsonl") as f:
        turns = [json.loads(line) for line in f]
    val = [t for t in turns if t["split"] == "validation"]
    rows = np.array([t["row"] for t in val])
    sub = wl.subset(rows)
    sp = split(Workload(tier.workload), int(cfg.get("tune_queries", 5000)),
               int(cfg.get("warmup", 2000)))
    target = float(cfg.get("target_recall", 0.95))
    retr = {}
    for m in ("cold", "trace"):
        L = tuned(Path(a.results), tier, ctx, m, sp, target)["L"]
        out = th.replay(build(ctx, m, L), np.ascontiguousarray(sub.Q), sub.sessions, k=10)
        retr[m] = out
    labels_all = np.concatenate([retr[m]["labels"][:, : a.k_ctx] for m in retr]).ravel()
    gt_rows = np.asarray(sub.gt_ids[:, : a.k_ctx]).ravel()
    passages = load_passages(a.passages, np.r_[labels_all, gt_rows])

    from vllm import LLM, SamplingParams

    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=8192)
    greedy = SamplingParams(temperature=0.0, max_tokens=64)
    one = SamplingParams(temperature=0.0, max_tokens=1)
    result, per_turn = {}, {}
    for name in ("exact", "cold", "trace"):
        prompts, hist = [], collections.defaultdict(list)
        for i, t in enumerate(val):
            if name == "exact":
                ids = sub.gt_ids[i, : a.k_ctx]
            else:
                ids = retr[name]["labels"][i, : a.k_ctx]
            ps = [passages[int(x)] for x in ids if int(x) >= 0]
            h = hist[t["conversation"]]
            prompts.append(prompt(h, t["question"], ps))
            h += [f"Q: {t['question']}", f"A: {t['answers'][0]}"]
        gens = llm.generate(prompts, greedy)
        scores = [f1(g.outputs[0].text, t["answers"]) for g, t in zip(gens, val)]
        per_turn[name] = np.asarray(scores)
        r = {"f1": 100.0 * float(np.mean(scores))}
        if name != "exact":
            out = retr[name]
            r["recall@5"] = float(recall_at_k(out["labels"], sub.gt_ids, a.k_ctx).mean())
            r["retrieval_p99_ms"] = float(np.percentile(out["wall_ns"] / 1e6, 99))
            pre = []
            for p in prompts[: a.ttft_turns]:  # prefill time, one request at a time
                t0 = time.perf_counter()
                llm.generate([p], one, use_tqdm=False)
                pre.append(time.perf_counter() - t0)
            ttft = np.asarray(pre) * 1e3 + out["wall_ns"][: len(pre)] / 1e6
            r["ttft_p50_ms"] = float(np.percentile(ttft, 50))
        result[name] = r
        print(name, r, flush=True)
    # TRACE minus cold, 95% interval from resampling whole conversations
    conv = np.array([t["conversation"] for t in val])
    groups = [np.flatnonzero(conv == c) for c in np.unique(conv)]
    diff = per_turn["trace"] - per_turn["cold"]
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(a.bootstrap):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        boots.append(100.0 * diff[idx].mean())
    result["f1_diff_trace_minus_cold"] = {
        "mean": 100.0 * float(diff.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "conversations": len(groups), "turns": int(len(val))}
    print("F1 difference", result["f1_diff_trace_minus_cold"], flush=True)
    save_json(Path(a.results) / "e2e" / "topiocqa.json", result)


if __name__ == "__main__":
    main()
