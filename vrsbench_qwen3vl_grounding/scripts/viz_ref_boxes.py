#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import cv2

JSON_RE = re.compile(r"\{[\s\S]*\}")


def load_jsonl_map(path: str, key: str) -> Dict[str, Any]:
    mp = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x = json.loads(line)
            k = x.get(key)
            if k is not None:
                mp[k] = x
    return mp


def parse_raw_bbox(raw_text: str) -> Optional[Tuple[float, float, float, float]]:
    """
    raw_text example: {"bbox":[700,131,976,343]}
    return (x1,y1,x2,y2) float in raw coord (0..1000)
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
            x1, y1, x2, y2 = [float(v) for v in bbox]
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            return x1, y1, x2, y2
    except Exception:
        return None
    return None


def raw1000_to_pixel(b_raw: Tuple[float, float, float, float], W: int, H: int) -> Tuple[int, int, int, int]:
    """
    raw coord in [0,1000] -> pixel coord
    """
    x1, y1, x2, y2 = b_raw
    px1 = int(round(x1 / 1000.0 * W))
    py1 = int(round(y1 / 1000.0 * H))
    px2 = int(round(x2 / 1000.0 * W))
    py2 = int(round(y2 / 1000.0 * H))

    # clamp
    px1 = max(0, min(W - 1, px1))
    px2 = max(0, min(W - 1, px2))
    py1 = max(0, min(H - 1, py1))
    py2 = max(0, min(H - 1, py2))

    if px1 > px2:
        px1, px2 = px2, px1
    if py1 > py2:
        py1, py2 = py2, py1
    return px1, py1, px2, py2


def draw_box(img, xyxy, color, label=None, thickness=2):
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        # label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def put_multiline_text(img, text: str, x: int, y: int, max_width: int):
    """
    Put wrapped text onto image (top-left x,y).
    """
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        tmp = (cur + " " + w).strip()
        (tw, _), _ = cv2.getTextSize(tmp, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        if tw <= max_width:
            cur = tmp
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    yy = y
    for ln in lines[:6]:  # cap lines to avoid huge overlay
        cv2.putText(img, ln, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        yy += 22


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_jsonl", default="data/ref_split_val.jsonl")
    ap.add_argument("--pred_jsonl", default="data/preds_qwen3vl_4b_val.jsonl")
    ap.add_argument("--out_dir", default="viz_out")
    ap.add_argument("--num", type=int, default=20, help="number of random samples to visualize")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample_id", type=str, default="", help="visualize a single sample_id")
    ap.add_argument("--only_have_pred", action="store_true", help="only sample items that have prediction raw bbox")
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load maps
    ref_map = load_jsonl_map(args.ref_jsonl, "sample_id")
    pred_map = load_jsonl_map(args.pred_jsonl, "sample_id")

    # Choose sample ids
    if args.sample_id:
        sids = [args.sample_id]
        if args.sample_id not in ref_map:
            raise SystemExit(f"sample_id not found in ref_jsonl: {args.sample_id}")
    else:
        all_ids = list(ref_map.keys())
        if args.only_have_pred:
            all_ids = [sid for sid in all_ids if sid in pred_map and isinstance(pred_map[sid].get("raw"), str)]
        if len(all_ids) == 0:
            raise SystemExit("No samples available after filtering.")
        sids = random.sample(all_ids, min(args.num, len(all_ids)))

    saved = 0
    for sid in sids:
        ref = ref_map[sid]
        img_path = ref["image_path"]
        expr = ref.get("expr", "")
        gt = ref.get("gt_box", None)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]

        # Draw GT
        if isinstance(gt, list) and len(gt) == 4:
            gx1, gy1, gx2, gy2 = [int(round(float(v))) for v in gt]
            gx1 = max(0, min(W - 1, gx1))
            gx2 = max(0, min(W - 1, gx2))
            gy1 = max(0, min(H - 1, gy1))
            gy2 = max(0, min(H - 1, gy2))
            if gx1 > gx2:
                gx1, gx2 = gx2, gx1
            if gy1 > gy2:
                gy1, gy2 = gy2, gy1
            draw_box(img, (gx1, gy1, gx2, gy2), (0, 255, 0), "GT")  # green

        # Draw Pred from raw (norm1000)
        pred = pred_map.get(sid, None)
        pred_xyxy = None
        raw_text = ""
        if pred is not None:
            raw_text = pred.get("raw", "") if isinstance(pred.get("raw", ""), str) else ""
            b_raw = parse_raw_bbox(raw_text)
            if b_raw is not None:
                px1, py1, px2, py2 = raw1000_to_pixel(b_raw, W, H)
                pred_xyxy = (px1, py1, px2, py2)
                draw_box(img, pred_xyxy, (0, 0, 255), "Pred")  # red

        # Put expr text on top
        # draw a dark box background
        cv2.rectangle(img, (0, 0), (W, 90), (0, 0, 0), -1)
        put_multiline_text(img, f"{sid} | {expr}", 10, 25, W - 20)

        # Also save raw json snippet
        if raw_text:
            cv2.rectangle(img, (0, 90), (W, 140), (0, 0, 0), -1)
            put_multiline_text(img, f"raw: {raw_text}", 10, 120, W - 20)

        out_path = out_dir / f"{sid}.png"
        cv2.imwrite(str(out_path), img)
        saved += 1

    print(f"[OK] saved {saved} visualizations to {out_dir}")


if __name__ == "__main__":
    main()