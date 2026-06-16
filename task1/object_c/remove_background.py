"""用 rembg 或 SAM 给图片去背景。"""

import argparse
import os
import sys
from pathlib import Path


def remove_background_rembg(input_path: str, output_path: str, model: str = "u2net"):
    try:
        from rembg import remove
        from PIL import Image
        import io
    except ImportError:
        print("缺 rembg，装一下: pip install rembg onnxruntime")
        sys.exit(1)

    print(f"读图: {input_path}")
    with open(input_path, "rb") as f:
        input_data = f.read()

    print(f"去背景中 (模型: {model})...")

    output_data = remove(
        input_data,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = Image.open(io.BytesIO(output_data))
    img.save(output_path, "PNG")
    print(f"去完背景 -> {output_path} ({img.size})")
    return output_path


def remove_background_sam(input_path: str, output_path: str, checkpoint: str = None):
    try:
        import torch
        import numpy as np
        from PIL import Image
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        print("缺 segment-anything: pip install git+https://github.com/facebookresearch/segment-anything.git")
        sys.exit(1)

    if checkpoint is None:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "models", "sam_vit_h_4b8939.pth"),
            os.path.expanduser("~/models/sam_vit_h_4b8939.pth"),
        ]
        checkpoint = next((p for p in candidates if os.path.exists(p)), candidates[0])

    if not os.path.exists(checkpoint):
        print(f"SAM 模型找不到: {checkpoint}")
        print("去 https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth 下载")
        sys.exit(1)

    print(f"加载 SAM: {checkpoint}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    image = np.array(Image.open(input_path).convert("RGB"))
    masks = mask_generator.generate(image)

    if not masks:
        print("SAM 啥也没检测到，换 rembg 试试")
        return remove_background_rembg(input_path, output_path)

    # 挑最可能是前景的 mask：优先覆盖中心，否则挑最近的
    h, w = image.shape[:2]
    cy, cx = h // 2, w // 2

    center_masks = [m for m in masks if m["segmentation"][cy, cx]]
    if center_masks:
        best_mask = max(center_masks, key=lambda x: x["area"])
    else:
        def bbox_center_dist(m):
            x, y, bw, bh = m["bbox"]
            bcy, bcx = y + bh / 2, x + bw / 2
            return (bcy - cy) ** 2 + (bcx - cx) ** 2
        best_mask = min(masks, key=bbox_center_dist)

    print(f"{len(masks)} 个物体中选了面积 {best_mask['area']} 的那个")

    mask = best_mask["segmentation"]
    result = Image.fromarray(image).convert("RGBA")
    result_array = np.array(result)
    result_array[:, :, 3] = (mask * 255).astype(np.uint8)
    result = Image.fromarray(result_array)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save(output_path, "PNG")
    print(f"去完背景 -> {output_path}")
    return output_path


def crop_to_bbox(input_path: str, output_path: str, padding: float = 0.1):
    from PIL import Image
    import numpy as np

    img = Image.open(input_path).convert("RGBA")
    img_array = np.array(img)

    # 找到非透明区域的边界
    alpha = img_array[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        print("图里没找到前景像素")
        return input_path

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # 添加 padding
    h, w = img_array.shape[:2]
    pad_h = int((rmax - rmin) * padding)
    pad_w = int((cmax - cmin) * padding)

    rmin = max(0, rmin - pad_h)
    rmax = min(h - 1, rmax + pad_h)
    cmin = max(0, cmin - pad_w)
    cmax = min(w - 1, cmax + pad_w)

    # 裁剪
    cropped = img.crop((cmin, rmin, cmax + 1, rmax + 1))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cropped.save(output_path, "PNG")

    print(f"裁切完 -> {output_path} ({img.size} -> {cropped.size})")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--method", type=str, default="rembg", choices=["rembg", "sam"])
    parser.add_argument("--model", type=str, default="u2net")
    parser.add_argument("--sam_checkpoint", type=str, default=None)
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--padding", type=float, default=0.1)
    args = parser.parse_args()

    if args.method == "rembg":
        output = remove_background_rembg(args.input, args.output, args.model)
    elif args.method == "sam":
        output = remove_background_sam(args.input, args.output, args.sam_checkpoint)

    if args.crop:
        crop_to_bbox(output, output, args.padding)
