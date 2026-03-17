#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import re
from collections import Counter
from rouge_score import rouge_scorer

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def tokenize(s: str):
    return TOKEN_RE.findall(s.lower())

def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(0, max(0, len(tokens)-n+1)))

def bleu_n(reference_tokens, hypothesis_tokens, n, smooth=1):
    """
    A simple corpus-level BLEU-n component:
      - modified precision for n-grams
      - brevity penalty
    This is not exactly sacrebleu, but stable and fully offline.
    """
    if len(hypothesis_tokens) == 0:
        return 0.0

    # brevity penalty
    ref_len = len(reference_tokens)
    hyp_len = len(hypothesis_tokens)
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(1, hyp_len))

    # modified precision
    ref_ngrams = ngram_counts(reference_tokens, n)
    hyp_ngrams = ngram_counts(hypothesis_tokens, n)

    overlap = 0
    total = 0
    for ng, c in hyp_ngrams.items():
        total += c
        overlap += min(c, ref_ngrams.get(ng, 0))

    # smoothing
    overlap += smooth
    total += smooth

    p = overlap / total
    return bp * p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap_jsonl", required=True)
    ap.add_argument("--pred_jsonl", required=True)
    args = ap.parse_args()

    gt = {x["sample_id"]: x["gt_caption"] for x in load_jsonl(args.cap_jsonl)}
    pred = {}
    for x in load_jsonl(args.pred_jsonl):
        sid = x.get("sample_id")
        if sid and isinstance(x.get("pred_caption"), str):
            pred[sid] = x["pred_caption"]

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    n = 0
    missing = 0
    b1=b2=b3=b4=0.0
    rL=0.0

    for sid, g in gt.items():
        p = pred.get(sid)
        if not p:
            missing += 1
            continue

        ref_tok = tokenize(g)
        hyp_tok = tokenize(p)

        b1 += bleu_n(ref_tok, hyp_tok, 1)
        b2 += bleu_n(ref_tok, hyp_tok, 2)
        b3 += bleu_n(ref_tok, hyp_tok, 3)
        b4 += bleu_n(ref_tok, hyp_tok, 4)

        rL += rouge.score(g, p)["rougeL"].fmeasure

        n += 1

    def avg(x): return x / max(1, n)

    print("\n==== Caption Eval (offline) ====")
    print(f"GT total: {len(gt)}")
    print(f"Pred available: {len(pred)}")
    print(f"Used (paired): {n}  Missing: {missing}")
    print({
        "BLEU-1(offline)": avg(b1),
        "BLEU-2(offline)": avg(b2),
        "BLEU-3(offline)": avg(b3),
        "BLEU-4(offline)": avg(b4),
        "ROUGE-L": avg(rL),
    })

if __name__ == "__main__":
    main()