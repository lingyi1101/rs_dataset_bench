import json
import argparse
from PIL import Image
import zipfile
import os
from io import BytesIO

def iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a + b - inter
    return 0.0 if union == 0 else inter / union

def to_pixel(box_0_1000, W, H):
    x1, y1, x2, y2 = box_0_1000
    return [x1 / 1000 * W, y1 / 1000 * H, x2 / 1000 * W, y2 / 1000 * H]

def open_image_from_zip(zip_path, image_name):
    """ 从zip文件中读取图像 """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(image_name) as img_file:
            img = Image.open(img_file)
            return img.convert("RGB")  # 确保转换为RGB模式

def open_image(image_path, zip_path=None):
    """ 尝试从解压的文件夹读取图像，如果文件不存在则从压缩文件读取 """
    try:
        img = Image.open(image_path)
        return img.convert("RGB")
    except FileNotFoundError:
        if zip_path:
            # 提取文件名并从压缩文件中读取图像
            image_name = os.path.basename(image_path)
            try:
                img = open_image_from_zip(zip_path, image_name)
                return img
            except KeyError:
                return None  # 如果压缩文件中也找不到该文件，返回 None
        return None  # 如果解压文件夹和压缩文件中都找不到图像，返回 None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--img_dir", required=True, help="The path to the folder containing images")
    ap.add_argument("--img_zip", required=True, help="The path to the ZIP file containing images")
    args = ap.parse_args()

    ious = []
    n = 0
    skipped = 0  # 记录跳过的图像数量

    with open(args.pred_jsonl, "r") as f:
        for line in f:
            rec = json.loads(line)
            pred = rec.get("pred_bbox_0_1000", None)
            if pred is None:
                continue
            gt = rec["gt_bbox"]

            # Get image file path and try to load the image
            image_path = rec["image"]
            img = open_image(image_path, zip_path=args.img_zip)

            if img is None:
                print(f"Warning: {image_path} not found in both folder and zip file, skipping...")
                skipped += 1
                continue  # Skip this image if not found

            W, H = img.size
            pred_px = to_pixel(pred, W, H)
            gt_px = gt

            v = iou(pred_px, gt_px)
            ious.append(v)
            n += 1

    if n == 0:
        print("No valid predictions.")
        return

    mean_iou = sum(ious) / n
    pr05 = sum(1 for x in ious if x >= 0.5) / n
    pr07 = sum(1 for x in ious if x >= 0.7) / n

    print(f"N={n}")
    print(f"Pr@0.5={pr05*100:.1f}")
    print(f"Pr@0.7={pr07*100:.1f}")
    print(f"meanIoU={mean_iou*100:.1f}")
    print(f"Skipped {skipped} images due to missing files.")

if __name__ == "__main__":
    main()
