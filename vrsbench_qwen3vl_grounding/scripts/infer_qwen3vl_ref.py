#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from typing import Set

from PIL import Image
from tqdm import tqdm

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

JSON_RE = re.compile(r"\{[\s\S]*\}")


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def parse_bbox_json(text: str, W: int, H: int):
    """
    Accept model output bbox as either:
      - pixel bbox: [0..W/H] (preferred)
      - normalized bbox: [0..1]

    Return normalized bbox01 in [0,1].
    """
    m = JSON_RE.search(text)
    if not m:
        return None
    s = m.group(0)
    try:
        obj = json.loads(s)
        bbox = obj.get("bbox", None)
        if not (isinstance(bbox, list) and len(bbox) == 4):
            return None

        x1, y1, x2, y2 = [float(v) for v in bbox]

        # If looks normalized, keep; else treat as pixels
        if max(x1, y1, x2, y2) <= 1.2:
            nx1, ny1, nx2, ny2 = x1, y1, x2, y2
        else:
            nx1, nx2 = x1 / max(1.0, W), x2 / max(1.0, W)
            ny1, ny2 = y1 / max(1.0, H), y2 / max(1.0, H)

        nx1, nx2 = sorted([clamp01(nx1), clamp01(nx2)])
        ny1, ny2 = sorted([clamp01(ny1), clamp01(ny2)])
        return [nx1, ny1, nx2, ny2]
    except Exception:
        return None


def load_done_ids(pred_path: str) -> Set[str]:
    done = set()
    if not pred_path or not os.path.exists(pred_path):
        return done
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                x = json.loads(line)
                sid = x.get("sample_id", None)
                if sid:
                    done.add(sid)
            except Exception:
                pass
    return done


def build_messages(expr: str, W: int, H: int, system_prompt: str):
    # 强约束：优先像素坐标，且明确范围；这样能明显减少输出 1000 的假设坐标
    user_text = (
        f"Referring expression: {expr}\n"
        f"Image size: width={W}, height={H}.\n"
        "Task: output the bounding box of the referred object.\n"
        "Output MUST be strict JSON only, no extra text.\n"
        "Format exactly: {\"bbox\":[x1,y1,x2,y2]}.\n"
        "IMPORTANT:\n"
        "- Prefer PIXEL coordinates.\n"
        f"- x must be in [0,{W}] and y must be in [0,{H}].\n"
        "- x1<x2 and y1<y2.\n"
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--ref_jsonl", type=str, default="data/ref_split_val.jsonl")
    ap.add_argument("--out", type=str, default="data/preds_qwen3vl_4b_val.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--dtype", type=str, default="bfloat16")  # bfloat16 / float16 / auto
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--raw_chars", type=int, default=800)
    args = ap.parse_args()

    done_ids = load_done_ids(args.out) if args.resume else set()
    if args.resume:
        print(f"[WORKER 0] resume: loaded {len(done_ids)} done ids from {args.out}")

    # 单卡：请用 CUDA_VISIBLE_DEVICES=9 指定物理卡
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=args.dtype, device_map={"": 0}
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model.eval()

    system_prompt = (
        "You are a visual grounding model.\n"
        "Given an image and a referring expression, output the bounding box of the referred object.\n"
        "You MUST respond with JSON ONLY. Do not output any other words. Do not repeat the prompt.\n"
        "Return best guess even if uncertain.\n"
    )

    out_mode = "a" if os.path.exists(args.out) else "w"
    n_run, n_skip = 0, 0

    with open(args.ref_jsonl, "r", encoding="utf-8") as fin, open(args.out, out_mode, encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Infer worker0"):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            sid = item["sample_id"]
            if args.resume and sid in done_ids:
                n_skip += 1
                continue

            img_path = item["image_path"]
            expr = item["expr"]

            img = Image.open(img_path).convert("RGB")
            W, H = img.size

            messages = build_messages(expr, W, H, system_prompt)
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                )

            # 只 decode 新生成部分
            prompt_len = inputs["input_ids"].shape[1]
            gen_ids = out_ids[:, prompt_len:]
            out_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

            bbox01 = parse_bbox_json(out_text, W, H)

            pred = {
                "sample_id": sid,
                "bbox01": bbox01,
                "raw": out_text[: args.raw_chars],
            }
            fout.write(json.dumps(pred, ensure_ascii=False) + "\n")
            fout.flush()

            n_run += 1
            if args.limit > 0 and n_run >= args.limit:
                break

    print("[DONE]")
    print(f"  ran: {n_run}")
    if args.resume:
        print(f"  skipped_done: {n_skip}")
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()