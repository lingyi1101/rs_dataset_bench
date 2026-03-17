#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_jsonl", type=str, required=True)
    ap.add_argument("--pred_jsonl", type=str, required=True)
    args = ap.parse_args()

    gt = {}
    with open(args.ref_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            gt[x["sample_id"]] = {
                "gt_box": x["gt_box"],
                "W": int(x["width"]),
                "H": int(x["height"]),
                "is_unique": bool(x.get("is_unique", False)),
            }

    preds = {}
    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            preds[x["sample_id"]] = x.get("bbox01", None)

    def eval_subset(filter_fn):
        n = 0
        invalid = 0
        hit05 = 0
        hit07 = 0
        ious = []

        for sid, g in gt.items():
            if not filter_fn(g):
                continue
            n += 1
            b01 = preds.get(sid, None)
            if b01 is None:
                invalid += 1
                ious.append(0.0)
                continue

            W, H = g["W"], g["H"]
            x1, y1, x2, y2 = b01
            pred_xyxy = [x1 * W, y1 * H, x2 * W, y2 * H]

            v = iou_xyxy(pred_xyxy, g["gt_box"])
            ious.append(v)
            if v >= 0.5:
                hit05 += 1
            if v >= 0.7:
                hit07 += 1

        return {
            "N": n,
            "invalid_or_missing": invalid,
            "Pr@0.5": 100.0 * hit05 / max(1, n),
            "Pr@0.7": 100.0 * hit07 / max(1, n),
            "meanIoU": sum(ious) / max(1, len(ious)),
        }

    all_res = eval_subset(lambda g: True)
    uniq_res = eval_subset(lambda g: g["is_unique"] is True)
    nonuniq_res = eval_subset(lambda g: g["is_unique"] is False)

    print("\n==== VRSBench Grounding (Referring) ====")
    print("[ALL]     ", all_res)
    print("[UNIQUE]  ", uniq_res)
    print("[NONUNIQ] ", nonuniq_res)


if __name__ == "__main__":
    main()