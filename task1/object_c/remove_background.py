"""
使用 rembg 或 SAM 对输入图像进行背景去除，获得纯净前景物体。

用法:
    python remove_background.py --input data/raw/photo.jpg --output data/processed/photo_nobg.png
"""

import argparse
import os
import sys
from pathlib import Path


def remove_background_rembg(input_path: str, output_path: str, model: str = "u2net"):
    """
    使用 rembg 库去除图像背景。
    支持的模型: u2net, u2netp, u2net_human_seg, silueta, isnet-general-use, isnet-anime
    """
    try:
        from rembg import remove
        from PIL import Image
        import io
    except ImportError:
        print("[错误] 请安装 rembg: pip install rembg onnxruntime")
        sys.exit(1)

    print(f"加载图像: {input_path}")
    with open(input_path, "rb") as f:
        input_data = f.read()

    print(f"使用模型: {model}")
    print("正在去除背景...")

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

    print(f"背景去除完成: {output_path}")
    print(f"  尺寸: {img.size}")
    print(f"  模式: {img.mode}")

    return output_path


def remove_background_sam(input_path: str, output_path: str, checkpoint: str = None):
    """
    使用 Segment Anything Model (SAM) 进行更精确的背景分割。
    适用于 rembg 效果不佳的复杂场景。
    """
    try:
        import torch
        import numpy as np
        from PIL import Image
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        print("[错误] 请安装 segment-anything:")
        print("  pip install git+https://github.com/facebookresearch/segment-anything.git")
        sys.exit(1)

    if checkpoint is None:
        # 优先查找项目本地 models 目录，其次 ~/models
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "models", "sam_vit_h_4b8939.pth"),
            os.path.expanduser("~/models/sam_vit_h_4b8939.pth"),
        ]
        checkpoint = next((p for p in candidates if os.path.exists(p)), candidates[0])

    if not os.path.exists(checkpoint):
        print(f"[错误] SAM 模型文件不存在: {checkpoint}")
        print("请下载: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        sys.exit(1)

    print(f"加载 SAM 模型: {checkpoint}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    print(f"加载图像: {input_path}")
    image = np.array(Image.open(input_path).convert("RGB"))

    print("正在生成分割掩码...")
    masks = mask_generator.generate(image)

    if not masks:
        print("[警告] 未检测到任何物体，使用 rembg 作为备选方案")
        return remove_background_rembg(input_path, output_path)

    # 选择最可能是前景的掩码：优先选覆盖图像中心的，其次选中心最近的
    h, w = image.shape[:2]
    cy, cx = h // 2, w // 2

    center_masks = [m for m in masks if m["segmentation"][cy, cx]]
    if center_masks:
        # 有掩码覆盖图像中心 → 取其中面积最大的
        best_mask = max(center_masks, key=lambda x: x["area"])
        reason = f"覆盖图像中心 (面积: {best_mask['area']})"
    else:
        # 没有掩码覆盖中心 → 取 bbox 中心离图像中心最近的
        def bbox_center_dist(m):
            x, y, bw, bh = m["bbox"]  # SAM bbox 格式: (x, y, w, h)
            bcy, bcx = y + bh / 2, x + bw / 2
            return (bcy - cy) ** 2 + (bcx - cx) ** 2
        best_mask = min(masks, key=bbox_center_dist)
        reason = f"最靠近图像中心 (面积: {best_mask['area']})"

    print(f"检测到 {len(masks)} 个物体，选择 {reason}")

    # 应用掩码
    mask = best_mask["segmentation"]
    result = Image.fromarray(image)
    result = result.convert("RGBA")
    result_array = np.array(result)
    result_array[:, :, 3] = (mask * 255).astype(np.uint8)
    result = Image.fromarray(result_array)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save(output_path, "PNG")

    print(f"背景去除完成: {output_path}")
    return output_path


def crop_to_bbox(input_path: str, output_path: str, padding: float = 0.1):
    """
    将图像裁剪到前景物体的边界框，并添加 padding。
    """
    from PIL import Image
    import numpy as np

    img = Image.open(input_path).convert("RGBA")
    img_array = np.array(img)

    # 找到非透明区域的边界
    alpha = img_array[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        print("[警告] 图像中没有前景像素")
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

    print(f"裁剪完成: {output_path}")
    print(f"  原始尺寸: {img.size}")
    print(f"  裁剪后尺寸: {cropped.size}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图像背景去除")
    parser.add_argument("--input", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output", type=str, required=True, help="输出图像路径")
    parser.add_argument("--method", type=str, default="rembg",
                        choices=["rembg", "sam"], help="背景去除方法")
    parser.add_argument("--model", type=str, default="u2net",
                        help="rembg 模型名称")
    parser.add_argument("--sam_checkpoint", type=str, default=None,
                        help="SAM 模型检查点路径")
    parser.add_argument("--crop", action="store_true",
                        help="去除背景后裁剪到边界框")
    parser.add_argument("--padding", type=float, default=0.1,
                        help="裁剪时的边界 padding 比例")
    args = parser.parse_args()

    if args.method == "rembg":
        output = remove_background_rembg(args.input, args.output, args.model)
    elif args.method == "sam":
        output = remove_background_sam(args.input, args.output, args.sam_checkpoint)

    if args.crop:
        crop_to_bbox(output, output, args.padding)
