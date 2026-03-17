import re, os, json, argparse
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile

from PIL import Image
import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

BOX_RE = re.compile(r"\[\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\]")

def parse_box(text):
    m = BOX_RE.search(text)
    if not m:
        return None
    return [int(m.group(i)) for i in range(1,5)]

def clamp(v, lo, hi): 
    return max(lo, min(hi, v))

def build_img_index(z: ZipFile):
    idx = {}
    for n in z.namelist():
        low = n.lower()
        if low.endswith((".jpg", ".jpeg", ".png")):
            idx[Path(n).name] = n
    return idx

def load_image(rec, img_zip: ZipFile, img_idx: dict):
    # 1) disk path
    p = rec.get("image")
    if p and os.path.exists(p):
        return Image.open(p).convert("RGB"), "disk"

    # 2) zip by filename (preferred)
    fname = rec.get("filename")
    if not fname:
        fname = Path(p).name if p else None
    if not fname:
        return None, "no_filename"

    full = img_idx.get(fname)
    if full is None:
        # extension / case fallback
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--img_zip", default="/netdisk/wanglu/whz/DIOR_RSVG/JPEGImages_fixed.zip")
    ap.add_argument("--max_samples", type=int, default=0, help="0 means all")
    args = ap.parse_args()

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model)

    records = [json.loads(x) for x in open(args.in_jsonl, "r")]
    if args.max_samples and args.max_samples > 0:
        records = records[:args.max_samples]

    img_zip_path = Path(args.img_zip)
    if not img_zip_path.exists():
        raise FileNotFoundError(f"image zip not found: {img_zip_path}")

    # counters
    total = 0
    ok = 0
    pred_parse_fail = 0
    infer_error = 0
    skip_img = 0
    skip_no_filename = 0
    skip_not_in_zip = 0
    img_from_disk = 0
    img_from_zip = 0

    with ZipFile(img_zip_path, "r") as z, open(args.out_jsonl, "w") as w:
        img_idx = build_img_index(z)

        for rec in tqdm(records, desc="infer"):
            total += 1

            # load image (disk -> zip)
            try:
                img, src = load_image(rec, z, img_idx)
            except Exception as e:
                img, src = None, f"load_exception:{type(e).__name__}"

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
            else:
                img_from_zip += 1

            query = (rec.get("query") or "").strip()
            if not query:
                # query 为空直接跳过：写入 error 方便统计
                rec["pred_text"] = ""
                rec["pred_bbox_0_1000"] = None
                rec["error"] = "empty_query"
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            prompt = (
                "You are doing referring visual grounding on a remote-sensing image.\n"
                f"Locate the object described as: \"{query}\".\n"
                "Return ONLY one bounding box in this exact format: [[x1, y1, x2, y2]].\n"
                "Use the 0-1000 relative coordinate system where (0,0) is top-left and (1000,1000) is bottom-right.\n"
                "Do NOT output any other text."
            )

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }]

            try:
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs.pop("token_type_ids", None)
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=64,
                        do_sample=False,
                        temperature=0.0,
                    )

                gen = out[0][inputs["input_ids"].shape[1]:]
                txt = processor.decode(gen, skip_special_tokens=True).strip()

            except Exception as e:
                infer_error += 1
                rec["pred_text"] = ""
                rec["pred_bbox_0_1000"] = None
                rec["error"] = f"infer_exception:{type(e).__name__}"
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            box = parse_box(txt)
            if box is not None:
                box = [clamp(x, 0, 1000) for x in box]
                ok += 1
                rec.pop("error", None)
            else:
                pred_parse_fail += 1
                rec["error"] = "pred_parse_fail"

            rec["pred_text"] = txt
            rec["pred_bbox_0_1000"] = box
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("==== Summary ====")
    print(f"total_samples           : {total}")
    print(f"ok_pred_bbox            : {ok}")
    print(f"pred_parse_fail          : {pred_parse_fail}")
    print(f"infer_error              : {infer_error}")
    print(f"skipped_missing_image     : {skip_img}")
    print(f"  - no_filename           : {skip_no_filename}")
    print(f"  - not_in_zip            : {skip_not_in_zip}")
    print(f"image_source_disk         : {img_from_disk}")
    print(f"image_source_zip          : {img_from_zip}")
    print(f"output_jsonl              : {args.out_jsonl}")

if __name__ == "__main__":
    main()
