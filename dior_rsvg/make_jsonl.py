# make_jsonl.py
import os, json, argparse
import xml.etree.ElementTree as ET
from pathlib import Path

def pick_first_text(node, candidates):
    for k in candidates:
        x = node.find(k)
        if x is not None and x.text:
            return x.text.strip()
    return None

def find_dir(root: Path, candidates):
    for c in candidates:
        p = root / c
        if p.exists() and p.is_dir():
            return p
    # fallback: one-level nested
    for name in candidates:
        hits = [h for h in root.glob(f"*/{name}") if h.is_dir()]
        if hits:
            return hits[0]
    return None

def resolve_xml_path(ann_dir: Path, _id: str) -> Path | None:
    """train.txt uses '3', but files are '00003.xml'."""
    stem = Path(_id).stem.strip()

    # 1) exact (in case already padded)
    p = ann_dir / f"{stem}.xml"
    if p.exists():
        return p
    p = ann_dir / f"{stem}.XML"
    if p.exists():
        return p

    # 2) numeric -> pad to 5 digits (matches your Annotations listing)
    if stem.isdigit():
        p2 = ann_dir / f"{int(stem):05d}.xml"
        if p2.exists():
            return p2
        p2 = ann_dir / f"{int(stem):05d}.XML"
        if p2.exists():
            return p2

    # 3) last resort fuzzy
    hits = list(ann_dir.glob(f"*{stem}*.xml"))
    if hits:
        return hits[0]

    return None

def parse_one_xml(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = pick_first_text(root, ["filename", "path"])
    if filename is None:
        raise ValueError(f"Cannot find filename in {xml_path}")

    items = []
    for obj in root.findall(".//object"):
        # DIOR-RSVG: expression is usually in <description>
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
        raise ValueError(f"No objects parsed from {xml_path}")

    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/netdisk/wanglu/whz/DIOR_RSVG")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    split_file = root / f"{args.split}.txt"
    ann_dir = find_dir(root, ["Annotations"])
    img_dir = find_dir(root, ["JPEGImages"])

    if ann_dir is None:
        raise FileNotFoundError(f"Annotations dir not found under: {root}")
    if img_dir is None:
        raise FileNotFoundError(f"JPEGImages dir not found under: {root}")
    if not split_file.exists():
        raise FileNotFoundError(f"split file not found: {split_file}")

    out_path = Path(args.out) if args.out else (root / f"dior_rsvg_{args.split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ids = [x.strip() for x in open(split_file) if x.strip()]

    n = 0
    missing_xml = 0
    empty_query = 0

    with open(out_path, "w") as w:
        for _id in ids:
            xml_path = resolve_xml_path(ann_dir, _id)
            if xml_path is None:
                missing_xml += 1
                print("[WARN] missing xml:", _id)
                continue

            try:
                items = parse_one_xml(str(xml_path))
            except Exception as e:
                print("[WARN]", _id, e)
                continue

            for j, it in enumerate(items):
                q = it["expression"]
                if not q.strip():
                    empty_query += 1

                rec = {
                    "id": f"{_id}_{j}",
                    "image": str(img_dir / it["filename"]),
                    "query": q,
                    "gt_bbox": it["bbox"],
                }
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1

    print(f"root={root}")
    print(f"ann_dir={ann_dir}")
    print(f"img_dir={img_dir}")
    print(f"split={args.split} ids={len(ids)} missing_xml={missing_xml}")
    print(f"done: {n} samples -> {out_path} (empty_query={empty_query})")

if __name__ == "__main__":
    main()
