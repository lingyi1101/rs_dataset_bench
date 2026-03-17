#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from typing import Dict, Any, Optional


def iou_xyxy(a, b) -> float:
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


def load_jsonl(path: str) -> list:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def safe_bbox01_to_xyxy(b01, W: int, H: int) -> Optional[list]:
    if not (isinstance(b01, list) and len(b01) == 4):
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in b01]
    except Exception:
        return None
    return [x1 * W, y1 * H, x2 * W, y2 * H]


def eval_subset(gt_map: Dict[str, Any], pred_map: Dict[str, Any], filter_fn, count_missing_as_zero: bool):
    """
    count_missing_as_zero:
      - False: 只在“有预测”的样本上计算（更适合 partial）
      - True : 缺失样本按 IoU=0 计入（lower bound）
    """
    n_gt = 0
    n_pred_avail = 0
    n_used = 0  # used in metric denominator (depends on mode)
    hit05 = 0
    hit07 = 0
    ious = []

    for sid, g in gt_map.items():
        if not filter_fn(g):
            continue
        n_gt += 1
        W, H = g["W"], g["H"]
        gt_xyxy = g["gt_box"]

        b01 = pred_map.get(sid, None)
        has_pred = b01 is not None
        if has_pred:
            n_pred_avail += 1

        if (not has_pred) and (not count_missing_as_zero):
            # partial mode: skip
            continue

        # now this sample contributes to metrics
        n_used += 1
        if not has_pred:
            ious.append(0.0)
            continue

        pred_xyxy = safe_bbox01_to_xyxy(b01, W, H)
        if pred_xyxy is None:
            ious.append(0.0)
            continue

        v = iou_xyxy(pred_xyxy, gt_xyxy)
        ious.append(v)
        if v >= 0.5:
            hit05 += 1
        if v >= 0.7:
            hit07 += 1

    mean_iou = sum(ious) / max(1, len(ious))
    pr05 = 100.0 * hit05 / max(1, n_used)
    pr07 = 100.0 * hit07 / max(1, n_used)
    coverage = 100.0 * n_pred_avail / max(1, n_gt)

    return {
        "GT_total": n_gt,
        "Pred_available": n_pred_avail,
        "Coverage%": coverage,
        "Denominator_used": n_used,
        "Pr@0.5": pr05,
        "Pr@0.7": pr07,
        "meanIoU": mean_iou,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_jsonl", required=True, help="data/ref_split_val.jsonl")
    ap.add_argument("--pred_jsonl", required=True, help="data/preds_qwen3vl_4b_val.jsonl (partial ok)")
    ap.add_argument("--missing_as_zero", action="store_true",
                    help="If set, treat missing preds as IoU=0 (lower bound). Default: evaluate only on available preds.")
    args = ap.parse_args()

    # GT
    gt_map = {}
    for x in load_jsonl(args.ref_jsonl):
        gt_map[x["sample_id"]] = {
            "gt_box": x["gt_box"],              # pixel xyxy
            "W": int(x["width"]),
            "H": int(x["height"]),
            "is_unique": bool(x.get("is_unique", False)),
        }

    # Pred (partial ok)
    pred_map = {}
    for x in load_jsonl(args.pred_jsonl):
        sid = x.get("sample_id")
        if not sid:
            continue
        b01 = x.get("bbox01", None)
        # 只收非空 bbox01；空的也可以收进来但等价于 missing
        if isinstance(b01, list) and len(b01) == 4:
            pred_map[sid] = b01

    mode = "LOWER_BOUND(missing=0)" if args.missing_as_zero else "PARTIAL_ONLY(on available preds)"
    print(f"\n==== VRSBench Grounding (Referring) Partial Eval ====")
    print(f"Mode: {mode}")
    print(f"GT samples: {len(gt_map)}")
    print(f"Pred samples (non-null bbox01): {len(pred_map)}")

    all_res = eval_subset(gt_map, pred_map, lambda g: True, args.missing_as_zero)
    uniq_res = eval_subset(gt_map, pred_map, lambda g: g["is_unique"] is True, args.missing_as_zero)
    nonuniq_res = eval_subset(gt_map, pred_map, lambda g: g["is_unique"] is False, args.missing_as_zero)

    print("\n[ALL]     ", all_res)
    print("[UNIQUE]  ", uniq_res)
    print("[NONUNIQ] ", nonuniq_res)

    print("\nTips:")
    print("- PARTIAL_ONLY 模式更适合你现在这种只跑了 2422 条的中间检查；指标只代表“已完成部分”的性能。")
    print("- missing_as_zero 模式给一个全量评测的下界（未跑到的都算 0 分）。")


if __name__ == "__main__":
    main()