# -*- coding: utf-8 -*-
"""
Qwen2-VL evaluation script for LRS-VQA

"""

import argparse
import json
import os
import re
from typing import Dict, Any, Iterable

import torch
from tqdm import tqdm
from PIL import Image, ImageFile

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# ===== PIL 安全设置（LRS-VQA 大图必备）=====
Image.MAX_IMAGE_PIXELS = 10_000_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ===== 工具函数 =====

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_done_ids(results_path: str) -> set:
    """读取已有结果文件，收集已完成的 id"""
    done = set()
    if not os.path.exists(results_path):
        return done

    with open(results_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] bad json at line {ln}, skip")
                continue
            sid = obj.get("id") or obj.get("question_id")
            if sid is not None:
                done.add(str(sid))
    return done


def resolve_image_path(img_field: str, image_root: str) -> str:
    """
    LRS_VQA/image/xxxx.tif
    image_root 通常是: .../LRS_VQA/image
    """

    base = os.path.basename(img_field)
    cand = os.path.join(image_root, base)
    if os.path.exists(cand):
        return cand

    cand2 = os.path.join(image_root, img_field)
    if os.path.exists(cand2):
        return cand2

    raise FileNotFoundError(f"Image not found: {img_field}")


def parse_sample(item: Dict[str, Any]):
    sid = str(item.get("question_id") or item.get("id"))
    question = item.get("text") or item.get("question")
    image = item.get("image")
    category = item.get("category", "unknown")
    gt = item.get("ground_truth", "")

    if sid is None or question is None or image is None:
        raise ValueError("Missing required fields in sample")

    return sid, image, question, category, gt


def build_prompt(q: str) -> str:
    q = q.strip()
    if "single word or phrase" not in q.lower():
        if not q.endswith("?"):
            q += "?"
        q += " Answer the question using a single word or phrase."
    return q


def postprocess_answer(ans: str) -> str:
    if not ans:
        return ""
    a = ans.strip().splitlines()[0]
    a = re.sub(r"^(answer\s*[:\-]\s*)", "", a, flags=re.I)
    a = a.strip(" \"'`")
    a = re.sub(r"[\.!\?]+$", "", a).strip()

    low = a.lower()
    if low.startswith("yes"):
        return "yes"
    if low.startswith("no"):
        return "no"
    return a


@torch.inference_mode()
def qwen2vl_generate(model, processor, image_path, prompt, device, max_new_tokens):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    gen = outputs[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True)


# ===== 主函数 =====

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--results_file", required=True)

    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=1280 * 28 * 28)

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)

    parser.add_argument("--resume", action="store_true",
                        help="resume from existing results_file, skip finished ids")

    args = parser.parse_args()

    # ===== 加载模型 =====
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    device = next(model.parameters()).device
    model.eval()

    # ===== 读取数据 =====
    samples = list(iter_jsonl(args.data_file))
    n = len(samples)

    start = max(0, args.start)
    end = n if args.end < 0 else min(args.end, n)

    # ===== 断点续跑 =====
    done_ids = set()
    if args.resume:
        done_ids = load_done_ids(args.results_file)
        print(f"[INFO] resume enabled, found {len(done_ids)} finished samples")

    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    mode = "a" if args.resume else "w"

    with open(args.results_file, mode, encoding="utf-8") as fout:
        for idx in tqdm(range(start, end), desc="Qwen2-VL Inference"):
            item = samples[idx]
            try:
                sid, img_field, question, category, gt = parse_sample(item)
                if args.resume and sid in done_ids:
                    continue

                image_path = resolve_image_path(img_field, args.image_root)
                prompt = build_prompt(question)

                raw = qwen2vl_generate(
                    model, processor,
                    image_path, prompt,
                    device, args.max_new_tokens
                )
                pred = postprocess_answer(raw)

            except Exception as e:
                print(f"[ERR] idx={idx}, id={item.get('question_id')}: {e}")
                pred = ""

            out = {
                "id": sid,
                "category": category,
                "ground_truth": gt,
                "text": pred,
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            if args.resume:
                done_ids.add(sid)


if __name__ == "__main__":
    main()
