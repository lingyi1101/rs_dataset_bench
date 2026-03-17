#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict

def pick_first_text(node, candidates):
    for k in candidates:
        x = node.find(k)
        if x is not None and x.text:
            return x.text.strip()
    return None

def parse_one_xml_bytes(xml_bytes: bytes):
    root = ET.fromstring(xml_bytes)

    filename = pick_first_text(root, ["filename", "path"])
    if filename is None:
        raise ValueError("Cannot find <filename> or <path> in xml")

    items = []
    for obj in root.findall(".//object"):
        # DIOR-RSVG 常见：<description>
        expr = pick_first_text(obj, ["description", "expression", "expr", "phrase", "refer", "ref", "text"])
        bb = obj.find("bndbox")
        if bb is None:
            continue

        x1 = int(float(pick_first_text(bb, ["xmin"]) or 0))
        y1 = int(float(pick_first_text(bb, ["ymin"]) or 0))
        x2 = int(float(pick_first_text(bb, ["xmax"]) or 0))
        y2 = int(float(pick_first_text(bb, ["ymax"]) or 0))

        if expr is None:
            expr = ""

        items.append({"filename": filename, "expression": expr, "bbox": [x1, y1, x2, y2]})

    if not items:
        raise ValueError("No objects parsed in xml")

    return items

def extract_5digit_key(s: str):
    """
    从字符串里提取 5 位数字 key（00003 这种）
    优先取 basename 里的 5 位；否则从全路径里找最后一个 5 位数字。
    """
    base = Path(s).name
    m = re.search(r'(\d{5})', base)
    if m:
        return m.group(1)
    ms = re.findall(r'(\d{5})', s)
    return ms[-1] if ms else None

def build_xml_index(z: ZipFile):
    """
    key(5位数字) -> [zip内xml路径...]
    避免 basename 重名覆盖的问题
    """
    idx = defaultdict(list)
    for n in z.namelist():
        if not n.lower().endswith(".xml"):
            continue
        key = extract_5digit_key(n)
        if key is not None:
            idx[key].append(n)
    return idx

def id_to_key(_id: str):
    stem = Path(_id).stem.strip()
    if stem.isdigit():
        return f"{int(stem):05d}"
    # fallback: 如果 split 里出现奇怪格式，也试着提取 5 位数字
    return extract_5digit_key(stem)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/netdisk/wanglu/whz/DIOR_RSVG")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--out", default=None, help="default: <root>/dior_rsvg_<split>.jsonl")
    ap.add_argument("--img_dir", default=None, help="default: <root>/JPEGImages (only used to form image path string)")
    ap.add_argument("--ann_zip", default=None, help="default: <root>/Annotations.zip")
    args = ap.parse_args()

    root = Path(args.root)
    split_file = root / f"{args.split}.txt"
    ann_zip = Path(args.ann_zip) if args.ann_zip else (root / "Annotations.zip")
    img_dir = Path(args.img_dir) if args.img_dir else (root / "JPEGImages")

    if not split_file.exists():
        raise FileNotFoundError(f"split file not found: {split_file}")
    if not ann_zip.exists():
        raise FileNotFoundError(f"Annotations.zip not found: {ann_zip}")

    out_path = Path(args.out) if args.out else (root / f"dior_rsvg_{args.split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ids = [x.strip() for x in open(split_file, "r") if x.strip()]

    n = 0
    missing_xml = 0
    empty_query = 0
    multi_path_hits = 0

    with ZipFile(ann_zip, "r") as z:
        idx = build_xml_index(z)

        with open(out_path, "w") as w:
            for _id in ids:
                key = id_to_key(_id)
                if not key or key not in idx:
                    missing_xml += 1
                    continue

                paths = idx[key]
                if len(paths) > 1:
                    # 同一个 key 对应多个路径，通常不影响，取第一个
                    multi_path_hits += 1

                try:
                    xml_bytes = z.read(paths[0])
                    items = parse_one_xml_bytes(xml_bytes)
                except Exception as e:
                    print("[WARN]", _id, key, "->", paths[0], e)
                    continue

                for j, it in enumerate(items):
                    q = it["expression"]
                    if not q.strip():
                        empty_query += 1

                    rec = {
                        "id": f"{_id}_{j}",
                        "image": str(img_dir / it["filename"]),  # 若不解压图片，这路径可不存在，但推理脚本可从zip取
                        "filename": it["filename"],              # 方便推理从 JPEGImages_fixed.zip 定位
                        "query": q,
                        "gt_bbox": it["bbox"],
                    }
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1

                    # 可选：每隔一段 flush，避免网盘缓冲导致你看不到增长
                    if n % 5000 == 0:
                        w.flush()

    print(f"root={root}")
    print(f"ann_zip={ann_zip}")
    print(f"img_dir={img_dir}")
    print(f"split={args.split} ids={len(ids)} missing_xml={missing_xml} multi_path_hits={multi_path_hits}")
    print(f"done: {n} samples -> {out_path} (empty_query={empty_query})")

if __name__ == "__main__":
    main()
