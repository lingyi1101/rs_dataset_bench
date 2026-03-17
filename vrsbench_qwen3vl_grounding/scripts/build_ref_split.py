#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm


IMG_EXTS = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"]


def corners_to_hbb(points: List[List[float]]) -> List[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def try_get_points(obj: Dict[str, Any]) -> Optional[List[List[float]]]:
    # 兼容四点/多点：obj_corner / corners / polygon / points
    for k in ["obj_corner", "corners", "polygon", "points"]:
        v = obj.get(k)
        if isinstance(v, list) and len(v) >= 4 and isinstance(v[0], (list, tuple)) and len(v[0]) >= 2:
            pts = [[float(p[0]), float(p[1])] for p in v]
            return pts
    return None


def try_get_bbox(obj: Dict[str, Any], W: int, H: int) -> Optional[List[float]]:
    """
    返回 pixel xyxy
    支持：
      - points(4+) -> HBB
      - xyxy: obj_coord / bbox / box / xyxy
      - xywh: bbox_xywh / xywh
    若 bbox 值疑似归一化(<=1.2)，自动乘回像素
    """
    pts = try_get_points(obj)
    if pts is not None:
        b = corners_to_hbb(pts)
        return b

    # 1) xyxy 类
    for k in ["obj_coord", "bbox", "box", "xyxy"]:
        v = obj.get(k)
        if isinstance(v, (list, tuple)) and len(v) == 4:
            x1, y1, x2, y2 = [float(v[0]), float(v[1]), float(v[2]), float(v[3])]
            # 归一化判断
            if max(x1, y1, x2, y2) <= 1.2:
                x1, x2 = x1 * W, x2 * W
                y1, y2 = y1 * H, y2 * H
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            return [x1, y1, x2, y2]

    # 2) xywh 类
    for k in ["bbox_xywh", "xywh"]:
        v = obj.get(k)
        if isinstance(v, (list, tuple)) and len(v) == 4:
            x, y, w, h = [float(v[0]), float(v[1]), float(v[2]), float(v[3])]
            if max(x, y, w, h) <= 1.2:
                x, w = x * W, w * W
                y, h = y * H, h * H
            return [x, y, x + w, y + h]

    return None


def try_get_refs(obj: Dict[str, Any]) -> List[str]:
    # 常见字段兼容
    for k in [
        "referring_sentence", "referring", "refs", "ref",
        "expressions", "expression", "caption", "text", "sentence"
    ]:
        v = obj.get(k)
        if not v:
            continue
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        if isinstance(v, list):
            out = []
            for s in v:
                if isinstance(s, str) and s.strip():
                    out.append(s.strip())
            if out:
                return out

    # fallback: 用类别拼一句
    cat = obj.get("category") or obj.get("name") or obj.get("label") or obj.get("obj_name") or "object"
    return [f"the {cat}"]


def find_objects(root: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    在不同标注结构里找 objects list
    """
    for k in ["objects", "annotations", "objs", "labels", "instances"]:
        v = root.get(k)
        if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict)):
            return v  # type: ignore
    return None


def find_image_by_stem(img_dir: Path, stem: str) -> Optional[Path]:
    # 先尝试直接同名不同后缀
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # 再递归搜（慢一点，但只做一次也行）
    for ext in IMG_EXTS:
        cand = next(img_dir.rglob(f"{stem}{ext}"), None)
        if cand is not None and cand.exists():
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dir", type=str, default="data/vrsbench_raw/val/Annotations_val")
    ap.add_argument("--img_dir", type=str, default="data/vrsbench_raw/val/Images_val")
    ap.add_argument("--out", type=str, default="data/ref_split_val.jsonl")
    ap.add_argument("--max_files", type=int, default=-1)
    ap.add_argument("--max_samples", type=int, default=-1)
    args = ap.parse_args()

    ann_dir = Path(args.ann_dir)
    img_dir = Path(args.img_dir)
    ann_files = sorted([p for p in ann_dir.glob("*.json")])

    if not ann_files:
        raise RuntimeError(f"No annotation json found under: {ann_dir}")
    if not img_dir.exists():
        raise RuntimeError(f"Image dir not found: {img_dir} (use --img_dir to set correct path)")

    n_written = 0
    n_noimg = 0
    n_noobj = 0
    n_nobox = 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for idx, ann_path in enumerate(tqdm(ann_files, desc="Building referring split (local val)")):
            if args.max_files > 0 and idx >= args.max_files:
                break

            stem = ann_path.stem  # e.g. 06491_0000
            img_path = find_image_by_stem(img_dir, stem)
            if img_path is None:
                n_noimg += 1
                continue

            with ann_path.open("r", encoding="utf-8") as f:
                root = json.load(f)

            # 读图尺寸
            try:
                with Image.open(img_path) as im:
                    W, H = im.size
            except Exception:
                n_noimg += 1
                continue

            objects = find_objects(root)
            if objects is None:
                n_noobj += 1
                continue

            for j, obj in enumerate(objects):
                if not isinstance(obj, dict):
                    continue

                gt_box = try_get_bbox(obj, W, H)
                if gt_box is None:
                    n_nobox += 1
                    continue

                refs = try_get_refs(obj)
                is_unique = bool(obj.get("is_unique", False))

                for k, expr in enumerate(refs):
                    if not expr.strip():
                        continue
                    sample = {
                        "sample_id": f"{stem}__obj{j}__ref{k}",
                        "image_id": stem,
                        "image_path": str(img_path),
                        "expr": expr,
                        "gt_box": gt_box,      # pixel xyxy
                        "width": int(W),
                        "height": int(H),
                        "is_unique": is_unique,
                    }
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    n_written += 1

                    if args.max_samples > 0 and n_written >= args.max_samples:
                        break
                if args.max_samples > 0 and n_written >= args.max_samples:
                    break
            if args.max_samples > 0 and n_written >= args.max_samples:
                break

    print(f"[OK] Saved {n_written} samples to {out_path}")
    print(f"[STAT] no_image={n_noimg}, no_objects={n_noobj}, no_box={n_nobox}")
    if n_written == 0:
        print("!!! Still 0 samples. 这说明 bbox/objects 的 key 可能都不在我兼容列表里。")
        print("    你可以执行：python -c 'import json;print(json.load(open(\"<one_ann.json>\")) )' 贴我一个样例，我再精准适配。")


if __name__ == "__main__":
    main()