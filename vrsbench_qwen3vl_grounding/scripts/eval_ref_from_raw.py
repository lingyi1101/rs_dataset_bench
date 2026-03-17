#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from typing import Dict, Any, Optional, Tuple

JSON_RE = re.compile(r"\{[\s\S]*\}")

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

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def order_xyxy(x1,y1,x2,y2):
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    return x1,y1,x2,y2

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def parse_raw_bbox(raw_text: str) -> Optional[Tuple[float,float,float,float]]:
    """
    raw_text like: {"bbox":[700,131,976,343]}
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None
    m = JSON_RE.search(raw_text)
    if not m:
        return None
    s = m.group(0)
    try:
        obj = json.loads(s)
        bbox = obj.get("bbox", None)
        if isinstance(bbox, list) and len(bbox) == 4:
            x1,y1,x2,y2 = [float(v) for v in bbox]
            return order_xyxy(x1,y1,x2,y2)
    except Exception:
        return None
    return None

def raw_to_pixel(b_raw, W: int, H: int, mode: str) -> Optional[Tuple[float,float,float,float]]:
    """
    Convert raw bbox into pixel xyxy according to mode:
      - pixel: raw is already pixels
      - norm1000: raw in [0,1000] coordinate system (x/1000*W, y/1000*H)
      - auto: detect
      - norm01: raw in [0,1] (rare)
    """
    if b_raw is None:
        return None
    x1,y1,x2,y2 = b_raw
    m = max(x1,y1,x2,y2)

    if mode == "auto":
        if m <= 1.2:
            mode2 = "norm01"
        elif m <= 1000.5:
            mode2 = "norm1000"
        else:
            mode2 = "pixel"
        mode = mode2

    if mode == "pixel":
        px1,py1,px2,py2 = x1,y1,x2,y2
    elif mode == "norm1000":
        px1,px2 = (x1/1000.0)*W, (x2/1000.0)*W
        py1,py2 = (y1/1000.0)*H, (y2/1000.0)*H
    elif mode == "norm01":
        px1,px2 = x1*W, x2*W
        py1,py2 = y1*H, y2*H
    else:
        raise ValueError(f"Unknown raw_mode: {mode}")

    px1,py1,px2,py2 = order_xyxy(px1,py1,px2,py2)

    # clamp to image bounds for fair IoU
    px1 = clamp(px1, 0.0, float(W))
    px2 = clamp(px2, 0.0, float(W))
    py1 = clamp(py1, 0.0, float(H))
    py2 = clamp(py2, 0.0, float(H))

    return (px1,py1,px2,py2)

def edge_or_degenerate(px, W: int, H: int, eps=1e-3):
    x1,y1,x2,y2 = px
    edge = (x1 <= eps) or (y1 <= eps) or (x2 >= W-eps) or (y2 >= H-eps)
    deg = ((x2-x1) <= eps) or ((y2-y1) <= eps)
    return edge, deg

def eval_subset(gt_map: Dict[str,Any], pred_raw_map: Dict[str,str], raw_mode: str, filter_fn):
    n_gt = 0
    n_pred = 0
    invalid = 0
    hit05 = 0
    hit07 = 0
    ious = []
    edge_cnt = 0
    deg_cnt = 0

    for sid, g in gt_map.items():
        if not filter_fn(g):
            continue
        n_gt += 1
        raw = pred_raw_map.get(sid, None)
        if raw is None:
            continue

        b_raw = parse_raw_bbox(raw)
        if b_raw is None:
            invalid += 1
            continue

        W,H = g["W"], g["H"]
        pred_px = raw_to_pixel(b_raw, W, H, raw_mode)
        if pred_px is None:
            invalid += 1
            continue

        n_pred += 1

        edge, deg = edge_or_degenerate(pred_px, W, H)
        edge_cnt += int(edge)
        deg_cnt += int(deg)

        gt_px = tuple(map(float, g["gt_box"]))
        v = iou_xyxy(pred_px, gt_px)
        ious.append(v)
        if v >= 0.5: hit05 += 1
        if v >= 0.7: hit07 += 1

    if n_pred == 0:
        return {
            "GT_total": n_gt,
            "Pred_available": 0,
            "Coverage%": 0.0,
            "Invalid": invalid,
            "Pr@0.5": 0.0,
            "Pr@0.7": 0.0,
            "meanIoU": 0.0,
            "Edge%": 0.0,
            "Degenerate%": 0.0,
        }

    return {
        "GT_total": n_gt,
        "Pred_available": n_pred,
        "Coverage%": 100.0 * n_pred / max(1, n_gt),
        "Invalid": invalid,
        "Pr@0.5": 100.0 * hit05 / max(1, n_pred),
        "Pr@0.7": 100.0 * hit07 / max(1, n_pred),
        "meanIoU": sum(ious) / max(1, len(ious)),
        "Edge%": 100.0 * edge_cnt / max(1, n_pred),
        "Degenerate%": 100.0 * deg_cnt / max(1, n_pred),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_jsonl", required=True, help="data/ref_split_val.jsonl (GT pixel + W/H)")
    ap.add_argument("--pred_jsonl", required=True, help="data/preds_*.jsonl (must include raw)")
    ap.add_argument("--raw_mode", default="auto", choices=["auto","pixel","norm1000","norm01"],
                    help="How to interpret raw bbox values")
    args = ap.parse_args()

    # GT map
    gt_map = {}
    for x in load_jsonl(args.ref_jsonl):
        gt_map[x["sample_id"]] = {
            "gt_box": x["gt_box"],     # pixel xyxy
            "W": int(x["width"]),
            "H": int(x["height"]),
            "is_unique": bool(x.get("is_unique", False)),
        }

    # Pred raw map
    pred_raw_map = {}
    for x in load_jsonl(args.pred_jsonl):
        sid = x.get("sample_id")
        raw = x.get("raw")
        if sid and isinstance(raw, str) and raw.strip():
            pred_raw_map[sid] = raw

    print("\n==== Eval from RAW bbox (alignment test) ====")
    print(f"raw_mode = {args.raw_mode}")
    print(f"GT total: {len(gt_map)}")
    print(f"Pred raw available: {len(pred_raw_map)} (partial ok)")

    all_res = eval_subset(gt_map, pred_raw_map, args.raw_mode, lambda g: True)
    uniq_res = eval_subset(gt_map, pred_raw_map, args.raw_mode, lambda g: g["is_unique"] is True)
    non_res = eval_subset(gt_map, pred_raw_map, args.raw_mode, lambda g: g["is_unique"] is False)

    print("\n[ALL]     ", all_res)
    print("[UNIQUE]  ", uniq_res)
    print("[NONUNIQ] ", non_res)

    print("\nNotes:")
    print("- 如果你怀疑 raw 是 0~1000 坐标系，请用 --raw_mode norm1000。")
    print("- Edge%/Degenerate% 高：说明预测框经常贴边或塌缩，IoU 会天然偏低。")

if __name__ == "__main__":
    main()