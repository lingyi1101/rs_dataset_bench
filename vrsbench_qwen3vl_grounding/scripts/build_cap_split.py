#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image
from tqdm import tqdm

IMG_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]

def find_image(img_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def find_caption(root: Dict[str, Any]) -> Optional[str]:
    # 常见 caption key
    for k in ["caption", "image_caption", "description", "dense_caption", "text"]:
        v = root.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # 有些放在更深层
    for k, v in root.items():
        if isinstance(v, dict):
            for kk in ["caption", "image_caption", "description", "dense_caption", "text"]:
                vv = v.get(kk)
                if isinstance(vv, str) and vv.strip():
                    return vv.strip()
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dir", default="data/vrsbench_raw/val/Annotations_val")
    ap.add_argument("--img_dir", default="data/vrsbench_raw/val/Images_val")
    ap.add_argument("--out", default="data/cap_split_val.jsonl")
    ap.add_argument("--max_files", type=int, default=-1)
    args = ap.parse_args()

    ann_dir = Path(args.ann_dir)
    img_dir = Path(args.img_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(ann_dir.glob("*.json"))
    n_written, n_noimg, n_nocap = 0, 0, 0

    with out_path.open("w", encoding="utf-8") as fout:
        for i, apath in enumerate(tqdm(ann_files, desc="Build caption split (val)")):
            if args.max_files > 0 and i >= args.max_files:
                break
            stem = apath.stem
            img_path = find_image(img_dir, stem)
            if img_path is None:
                n_noimg += 1
                continue

            root = json.load(open(apath, "r", encoding="utf-8"))
            cap = find_caption(root)
            if cap is None:
                n_nocap += 1
                continue

            with Image.open(img_path) as im:
                W, H = im.size

            item = {
                "sample_id": f"{stem}__cap0",
                "image_id": stem,
                "image_path": str(img_path),
                "gt_caption": cap,
                "width": int(W),
                "height": int(H),
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[OK] wrote {n_written} caption samples -> {out_path}")
    print(f"[STAT] no_image={n_noimg}, no_caption={n_nocap}")

if __name__ == "__main__":
    main()