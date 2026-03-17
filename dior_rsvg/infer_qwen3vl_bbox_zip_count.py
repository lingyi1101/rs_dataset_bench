import os, json, argparse, re
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

def build_img_index(z: ZipFile):
    idx = {}
    for n in z.namelist():
        low = n.lower()
        if low.endswith((".jpg", ".jpeg", ".png")):
            idx[Path(n).name] = n
    return idx

def load_image(rec, img_zip: ZipFile, img_idx: dict):
    # 1) 本地路径优先（如果你部分解压了也能复用）
    p = rec.get("image")
    if p and os.path.exists(p):
        return Image.open(p).convert("RGB"), "disk"

    # 2) 从 zip 读取（优先 filename 字段）
    fname = rec.get("filename")
    if not fname:
        fname = Path(p).name if p else None
    if not fname:
        return None, "no_filename"

    full = img_idx.get(fname)
    if full is None:
        # 扩展名大小写兜底
        stem = Path(fname).stem
        for ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
            cand = stem + ext
            if cand in img_idx:
                full = img_idx[cand]
                break
    if full is None:
        return None, "not_in_zip"

    b = img_zip.read(full)
    return Image.open(BytesIO(b)).convert("RGB"), "zip"

def extract_coords(text):
    bracket = re.findall(r'\[([\d,\s\.]+)\]', text)
    vals = []
    if bracket:
        for m in bracket:
            nums = re.findall(r'\d+(?:\.\d+)?', m)
            vals.extend([float(x) for x in nums])
    else:
        vals = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', text)]
    return vals

def parse_pred_box_1000(text):
    vals = extract_coords(text)
    if len(vals) < 4:
        return None
    x1, y1, x2, y2 = vals[0], vals[1], vals[2], vals[3]
    # clamp to [0,1000]
    x1 = max(0, min(1000, x1)); y1 = max(0, min(1000, y1))
    x2 = max(0, min(1000, x2)); y2 = max(0, min(1000, y2))
    return [x1, y1, x2, y2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--model", default="/root/autodl-tmp/Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--img_zip", default="/netdisk/wanglu/whz/DIOR_RSVG/JPEGImages_fixed.zip")
    ap.add_argument("--max_samples", type=int, default=0, help="0 means all")
    args = ap.parse_args()

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = [json.loads(x) for x in open(in_path, "r")]
    if args.max_samples and args.max_samples > 0:
        records = records[:args.max_samples]

    img_zip_path = Path(args.img_zip)
    if not img_zip_path.exists():
        raise FileNotFoundError(f"image zip not found: {img_zip_path}")

    # 统计计数器
    total = 0
    ok = 0
    img_from_disk = 0
    img_from_zip = 0
    skip_img = 0
    skip_no_filename = 0
    skip_not_in_zip = 0
    infer_error = 0
    pred_parse_fail = 0

    with ZipFile(img_zip_path, "r") as z:
        img_idx = build_img_index(z)

        with open(out_path, "w") as w:
            for rec in tqdm(records, desc="infer"):
                total += 1

                # 读图：允许缺图则跳过，但写出一条带 error 的记录
                try:
                    img, src = load_image(rec, z, img_idx)
                except Exception as e:
                    img = None
                    src = f"load_exception:{type(e).__name__}"

                if img is None:
                    skip_img += 1
                    if src == "no_filename":
                        skip_no_filename += 1
                        err = "image_missing_no_filename"
                    elif src == "not_in_zip":
                        skip_not_in_zip += 1
                        err = "image_missing_not_in_zip"
                    else:
                        err = f"image_missing:{src}"

                    rec["pred_text"] = ""
                    rec["pred_bbox_0_1000"] = None
                    rec["error"] = err
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue

                if src == "disk":
                    img_from_disk += 1
                elif src == "zip":
                    img_from_zip += 1

                query = rec.get("query", "")
                prompt = (
                    "<image>\n"
                    f"Please locate the object described as: {query}\n"
                    "Return ONLY one bounding box in the format [[x1,y1,x2,y2]] "
                    "where coordinates are integers in [0,1000] (relative to image)."
                )

                try:
                    inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        out = model.generate(
                            **inputs,
                            max_new_tokens=64,
                            do_sample=False,
                        )
                    text = processor.batch_decode(out, skip_special_tokens=True)[0]
                except Exception as e:
                    infer_error += 1
                    rec["pred_text"] = ""
                    rec["pred_bbox_0_1000"] = None
                    rec["error"] = f"infer_exception:{type(e).__name__}"
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue

                pred = parse_pred_box_1000(text)
                if pred is None:
                    pred_parse_fail += 1
                    rec["error"] = "pred_parse_fail"
                else:
                    rec.pop("error", None)
                    ok += 1

                rec["pred_text"] = text
                rec["pred_bbox_0_1000"] = pred
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 总结打印（关键：告诉你跳过多少）
    print("==== Summary ====")
    print(f"total_samples          : {total}")
    print(f"ok_pred_bbox           : {ok}")
    print(f"pred_parse_fail         : {pred_parse_fail}")
    print(f"infer_error             : {infer_error}")
    print(f"skipped_missing_image    : {skip_img}")
    print(f"  - no_filename          : {skip_no_filename}")
    print(f"  - not_in_zip           : {skip_not_in_zip}")
    print(f"image_source_disk        : {img_from_disk}")
    print(f"image_source_zip         : {img_from_zip}")
    print(f"output_jsonl             : {out_path}")

if __name__ == "__main__":
    main()
