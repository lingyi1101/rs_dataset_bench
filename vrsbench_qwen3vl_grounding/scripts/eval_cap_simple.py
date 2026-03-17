#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from collections import defaultdict

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def tokenize(s: str):
    return nltk.word_tokenize(s.lower())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap_jsonl", required=True)
    ap.add_argument("--pred_jsonl", required=True)
    args = ap.parse_args()

    gt = {}
    for x in load_jsonl(args.cap_jsonl):
        gt[x["sample_id"]] = x["gt_caption"]

    pred = {}
    for x in load_jsonl(args.pred_jsonl):
        if "sample_id" in x and "pred_caption" in x:
            pred[x["sample_id"]] = x["pred_caption"]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method3

    bleu1=bleu2=bleu3=bleu4=0.0
    rougeL=0.0
    meteor=0.0
    n=0
    missing=0

    for sid, g in gt.items():
        p = pred.get(sid)
        if not p:
            missing += 1
            continue
        ref_tok = [tokenize(g)]
        hyp_tok = tokenize(p)

        bleu1 += sentence_bleu(ref_tok, hyp_tok, weights=(1,0,0,0), smoothing_function=smooth)
        bleu2 += sentence_bleu(ref_tok, hyp_tok, weights=(0.5,0.5,0,0), smoothing_function=smooth)
        bleu3 += sentence_bleu(ref_tok, hyp_tok, weights=(1/3,1/3,1/3,0), smoothing_function=smooth)
        bleu4 += sentence_bleu(ref_tok, hyp_tok, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)

        rougeL += scorer.score(g, p)["rougeL"].fmeasure
        meteor += meteor_score([g], p)

        n += 1

    def avg(x): return x / max(1, n)

    print("\n==== Caption Eval (simple) ====")
    print(f"GT total: {len(gt)}")
    print(f"Pred available: {len(pred)}")
    print(f"Used (paired): {n}  Missing: {missing}")
    print({
        "BLEU-1": avg(bleu1),
        "BLEU-2": avg(bleu2),
        "BLEU-3": avg(bleu3),
        "BLEU-4": avg(bleu4),
        "ROUGE-L": avg(rougeL),
        "METEOR": avg(meteor),
    })

if __name__ == "__main__":
    main()