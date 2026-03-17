#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from collections import Counter

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def normalize(s: str) -> str:
    s = s.lower().strip()
    # remove punctuation
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def token_f1(a: str, b: str) -> float:
    ta = a.split()
    tb = b.split()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    ca = Counter(ta)
    cb = Counter(tb)
    common = sum((ca & cb).values())
    if common == 0:
        return 0.0
    p = common / len(tb)
    r = common / len(ta)
    return 2*p*r/(p+r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqa_jsonl", required=True)
    ap.add_argument("--pred_jsonl", required=True)
    args = ap.parse_args()

    gt = {x["sample_id"]: x["gt_answer"] for x in load_jsonl(args.vqa_jsonl)}
    pred = {x["sample_id"]: x["pred_answer"] for x in load_jsonl(args.pred_jsonl) if "pred_answer" in x}

    n = 0
    exact = 0
    f1_sum = 0.0
    missing = 0

    for sid, ga in gt.items():
        pa = pred.get(sid)
        if pa is None:
            missing += 1
            continue
        g = normalize(ga)
        p = normalize(pa)
        n += 1
        if g == p:
            exact += 1
        f1_sum += token_f1(g, p)

    print("\n==== VQA Eval (simple) ====")
    print(f"GT total: {len(gt)}")
    print(f"Pred available: {len(pred)}")
    print(f"Used (paired): {n}  Missing: {missing}")
    print({
        "ExactMatch%": 100.0 * exact / max(1, n),
        "TokenF1": f1_sum / max(1, n),
    })

if __name__ == "__main__":
    main()