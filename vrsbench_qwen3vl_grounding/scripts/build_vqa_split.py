#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

IMG_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]

def find_image(img_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def find_qas(root: Dict[str, Any]) -> List[Dict[str, Any]]:
    # 常见字段：qas / qa / questions / vqa / conversations
    for k in ["qas", "qa", "vqa", "questions", "question_answer_pairs"]:
        v = root.get(k)
        if isinstance(v, list) and (len(v)==0 or isinstance(v[0], dict)):
            return v
    # deeper
    for _, v in root.items():
        if isinstance(v, dict):
            for k in ["qas", "qa", "vqa", "questions", "question_answer_pairs"]:
                vv = v.get(k)
                if isinstance(vv, list) and (len(vv)==0 or isinstance(vv[0], dict)):
                    return vv
    return []

def get_qa_fields(qa: Dict[str, Any]) -> Optional[Tuple[str,str]]:
    # question keys
    q = None
    for k in ["question", "q", "query", "prompt"]:
        if isinstance(qa.get(k), str) and qa[k].strip():
            q = qa[k].strip()
            break
    # answer keys
    a = None
    for k in ["answer", "a", "gt", "label", "response"]:
        if isinstance(qa.get(k), str) and qa[k].strip():
            a = qa[k].strip()
            break
    if q and a:
        return q, a
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dir", default="data/vrsbench_raw/val/Annotations_val")
    ap.add_argument("--img_dir", default="data/vrsbench_raw/val/Images_val")
    ap.add_argument("--out", default="data/vqa_split_val.jsonl")
    ap.add_argument("--max_files", type=int, default=-1)
    ap.add_argument("--max_qas_per_image", type=int, default=-1)
    args = ap.parse_args()

    ann_dir = Path(args.ann_dir)
    img_dir = Path(args.img_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(ann_dir.glob("*.json"))
    n_written, n_noimg, n_noqa = 0, 0, 0

    with out_path.open("w", encoding="utf-8") as fout:
        for i, apath in enumerate(tqdm(ann_files, desc="Build VQA split (val)")):
            if args.max_files > 0 and i >= args.max_files:
                break
            stem = apath.stem
            img_path = find_image(img_dir, stem)
            if img_path is None:
                n_noimg += 1
                continue

            root = json.load(open(apath, "r", encoding="utf-8"))
            qas = find_qas(root)
            if not qas:
                n_noqa += 1
                continue

            with Image.open(img_path) as im:
                W, H = im.size

            cnt_img = 0
            for j, qa in enumerate(qas):
                if not isinstance(qa, dict):
                    continue
                pair = get_qa_fields(qa)
                if pair is None:
                    continue
                q, a = pair
                item = {
                    "sample_id": f"{stem}__qa{j}",
                    "image_id": stem,
                    "image_path": str(img_path),
                    "question": q,
                    "gt_answer": a,
                    "width": int(W),
                    "height": int(H),
                }
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                n_written += 1
                cnt_img += 1
                if args.max_qas_per_image > 0 and cnt_img >= args.max_qas_per_image:
                    break

    print(f"[OK] wrote {n_written} VQA samples -> {out_path}")
    print(f"[STAT] no_image={n_noimg}, no_vqa={n_noqa}")

if __name__ == "__main__":
    main()