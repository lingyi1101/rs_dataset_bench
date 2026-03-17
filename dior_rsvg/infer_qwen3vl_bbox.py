# infer_qwen3vl_bbox.py
import re, json, argparse
from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

BOX_RE = re.compile(r"\[\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\]")

def parse_box(text):
    m = BOX_RE.search(text)
    if not m:
        return None
    box = [int(m.group(i)) for i in range(1,5)]
    return box

def clamp(v, lo, hi): return max(lo, min(hi, v))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model)

    with open(args.in_jsonl, "r") as f, open(args.out_jsonl, "w") as w:
        for line in f:
            rec = json.loads(line)
            img = Image.open(rec["image"]).convert("RGB")

            query = rec["query"].strip()
            if not query:
                # 若你的 XML parser 没抓到 query，这里先跳过，避免影响指标
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

            box = parse_box(txt)
            if box is not None:
                # 兼容一些“坐标系漂移”：统一 clamp 到 0..1000
                box = [clamp(x, 0, 1000) for x in box]

            rec["pred_text"] = txt
            rec["pred_bbox_0_1000"] = box
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("done ->", args.out_jsonl)

if __name__ == "__main__":
    main()
