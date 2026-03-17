#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Set

from PIL import Image
from tqdm import tqdm

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

def load_done_ids(path: str) -> Set[str]:
    done = set()
    if not path or not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                x = json.loads(line)
                sid = x.get("sample_id")
                if sid:
                    done.add(sid)
            except Exception:
                pass
    return done

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--vqa_jsonl", default="data/vqa_split_val.jsonl")
    ap.add_argument("--out", default="data/vqa_preds_qwen3vl_4b_val.jsonl")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    done = load_done_ids(args.out) if args.resume else set()
    if args.resume:
        print(f"[RESUME] loaded {len(done)} done ids from {args.out}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=args.dtype, device_map={"": 0}
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model.eval()

    system = (
        "You are a remote-sensing VQA assistant. Answer the question based on the image. "
        "Be concise. Output the answer only (no explanation)."
    )

    out_mode = "a" if os.path.exists(args.out) else "w"
    n = 0
    with open(args.vqa_jsonl, "r", encoding="utf-8") as fin, open(args.out, out_mode, encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Infer VQA"):
            x = json.loads(line)
            sid = x["sample_id"]
            if args.resume and sid in done:
                continue

            img = Image.open(x["image_path"]).convert("RGB")
            q = x["question"]

            messages = [
                {"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
            ]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out_ids = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)

            gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
            ans = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

            fout.write(json.dumps({"sample_id": sid, "pred_answer": ans}, ensure_ascii=False) + "\n")
            fout.flush()

            n += 1
            if args.limit > 0 and n >= args.limit:
                break

    print(f"[OK] saved -> {args.out}")

if __name__ == "__main__":
    main()