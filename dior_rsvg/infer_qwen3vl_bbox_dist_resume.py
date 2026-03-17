import os, re, json, argparse
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile

import torch
import torch.distributed as dist
from PIL import Image
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
    p = rec.get("image")
    if p and os.path.exists(p):
        return Image.open(p).convert("RGB"), "disk"

    fname = rec.get("filename")
    if not fname:
        fname = Path(p).name if p else None
    if not fname:
        return None, "no_filename"

    full = img_idx.get(fname)
    if full is None:
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

def setup_dist():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank

def barrier(world):
    ##if world > 1:
        ##dist.barrier()
    return

def load_done_ids(out_shard: Path):
    done = set()
    if not out_shard.exists():
        return done
    with open(out_shard, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                _id = rec.get("id")
                if _id:
                    done.add(_id)
            except Exception:
                continue
    return done

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--img_zip", default="/netdisk/wanglu/whz/DIOR_RSVG/JPEGImages_fixed.zip")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    rank, world, local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map={"": local_rank},
            attn_implementation="sdpa",
        )
        processor = AutoProcessor.from_pretrained(args.model)

        in_path = Path(args.in_jsonl)
        out_path = Path(args.out_jsonl)
        out_shard = Path(str(out_path) + f".rank{rank}.jsonl")
        stat_path = Path(str(out_path) + f".rank{rank}.stats.json")

        img_zip_path = Path(args.img_zip)
        if not img_zip_path.exists():
            raise FileNotFoundError(f"image zip not found: {img_zip_path}")

        done_ids = set()
        if args.resume:
            done_ids = load_done_ids(out_shard)
            print(f"[resume] rank{rank} loaded done_ids={len(done_ids)} from {out_shard}")

        seen = 0
        skipped_resume = 0
        processed = 0
        ok = 0
        pred_parse_fail = 0
        infer_error = 0
        skip_img = 0
        skip_no_filename = 0
        skip_not_in_zip = 0
        img_from_disk = 0
        img_from_zip = 0
        empty_query = 0

        out_mode = "a" if args.resume else "w"

        with ZipFile(img_zip_path, "r") as z, open(in_path, "r") as f, open(out_shard, out_mode) as w:
            img_idx = build_img_index(z)
            pbar = tqdm(desc=f"infer world={world} rank0", total=None) if rank == 0 else None

            for idx, line in enumerate(f):
                if args.max_samples and idx >= args.max_samples:
                    break
                if (idx % world) != rank:
                    continue

                rec = json.loads(line)
                rec_id = rec.get("id")
                seen += 1

                if args.resume and rec_id and rec_id in done_ids:
                    skipped_resume += 1
                    if pbar is not None: pbar.update(1)
                    continue

                rec["_orig_idx"] = idx

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
                    processed += 1
                    if pbar is not None: pbar.update(1)
                    continue

                if src == "disk":
                    img_from_disk += 1
                else:
                    img_from_zip += 1

                query = (rec.get("query") or "").strip()
                if not query:
                    empty_query += 1
                    rec["pred_text"] = ""
                    rec["pred_bbox_0_1000"] = None
                    rec["error"] = "empty_query"
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    processed += 1
                    if pbar is not None: pbar.update(1)
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
                    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

                    with torch.no_grad():
                        out = model.generate(
                            **inputs,
                            max_new_tokens=64,
                            do_sample=False,
                        )
                    gen = out[0][inputs["input_ids"].shape[1]:]
                    txt = processor.decode(gen, skip_special_tokens=True).strip()

                except Exception as e:
                    infer_error += 1
                    rec["pred_text"] = ""
                    rec["pred_bbox_0_1000"] = None
                    rec["error"] = f"infer_exception:{type(e).__name__}"
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    processed += 1
                    if pbar is not None: pbar.update(1)
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
                processed += 1
                if pbar is not None: pbar.update(1)

            if pbar is not None:
                pbar.close()

        stats = {
            "rank": rank,
            "world": world,
            "seen_shard_lines": seen,
            "skipped_resume": skipped_resume,
            "processed_new": processed,
            "ok_pred_bbox": ok,
            "pred_parse_fail": pred_parse_fail,
            "infer_error": infer_error,
            "skipped_missing_image": skip_img,
            "skip_no_filename": skip_no_filename,
            "skip_not_in_zip": skip_not_in_zip,
            "empty_query": empty_query,
            "image_source_disk": img_from_disk,
            "image_source_zip": img_from_zip,
            "out_shard": str(out_shard),
        }
        with open(stat_path, "w") as sf:
            json.dump(stats, sf, ensure_ascii=False, indent=2)

        barrier(world)

        if args.merge and rank == 0:
            shards = [Path(str(out_path) + f".rank{r}.jsonl") for r in range(world)]
            all_recs = []
            for sp in shards:
                if not sp.exists():
                    print("[WARN] missing shard:", sp)
                    continue
                with open(sp, "r") as f:
                    for line in f:
                        try:
                            all_recs.append(json.loads(line))
                        except Exception:
                            continue

            all_recs.sort(key=lambda x: x.get("_orig_idx", 10**18))

            with open(out_path, "w") as w:
                for rec in all_recs:
                    rec.pop("_orig_idx", None)
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")

            total_sum = {}
            for r in range(world):
                stp = Path(str(out_path) + f".rank{r}.stats.json")
                if not stp.exists():
                    continue
                s = json.load(open(stp, "r"))
                for k, v in s.items():
                    if isinstance(v, int):
                        total_sum[k] = total_sum.get(k, 0) + v

            print("==== Global Summary (summed ints) ====")
            for k in [
                "seen_shard_lines","skipped_resume","processed_new",
                "ok_pred_bbox","pred_parse_fail","infer_error",
                "skipped_missing_image","skip_no_filename","skip_not_in_zip",
                "empty_query","image_source_disk","image_source_zip"
            ]:
                if k in total_sum:
                    print(f"{k:22s}: {total_sum[k]}")
            print("merged_out_jsonl        :", out_path)

        barrier(world)

    finally:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1 and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
