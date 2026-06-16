"""
从环绕视频中按固定间隔抽取帧图像，作为 COLMAP 的输入。
用法:
    python extract_frames.py --video path/to/video.mp4 --output_dir data/images --interval 2
"""

import argparse
import os
import cv2
from pathlib import Path


def extract_frames(video_path: str, output_dir: str, interval: int = 2,
                   max_frames: int = 300, max_size: int = None):
    """从视频中每隔 interval 帧抽取一帧图像"""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_size is not None:
        scale = max_size / max(w, h)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            print(f"视频信息: {total_frames} 帧, {fps:.1f} FPS, 时长 {total_frames/fps:.1f}s, "
                  f"分辨率 {w}x{h} -> {new_w}x{new_h}")
        else:
            new_w, new_h = w, h
            print(f"视频信息: {total_frames} 帧, {fps:.1f} FPS, 时长 {total_frames/fps:.1f}s, "
                  f"分辨率 {w}x{h} (无需缩放)")
    else:
        new_w, new_h = w, h
        print(f"视频信息: {total_frames} 帧, {fps:.1f} FPS, 时长 {total_frames/fps:.1f}s")

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            if new_w != w:
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            out_path = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1

            if saved_count >= max_frames:
                print(f"已达到最大帧数 {max_frames}，停止抽取")
                break

        frame_idx += 1

    cap.release()
    print(f"共抽取 {saved_count} 帧图像 -> {output_dir}")
    return saved_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从视频中抽取帧图像")
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--output_dir", type=str, default="data/images", help="输出图像目录")
    parser.add_argument("--interval", type=int, default=2, help="每隔多少帧抽取一帧")
    parser.add_argument("--max_frames", type=int, default=300, help="最大抽取帧数")
    parser.add_argument("--max_size", type=int, default=None,
                        help="图像最大边长，超过则等比缩放 (如 1280)")
    args = parser.parse_args()

    extract_frames(args.video, args.output_dir, args.interval, args.max_frames, args.max_size)
