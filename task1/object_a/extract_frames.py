"""从视频抽帧，给 COLMAP 用。"""

import argparse
import os
import cv2


def extract_frames(video_path: str, output_dir: str, interval: int = 2,
                   max_frames: int = 300, max_size: int = None):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    new_w, new_h = w, h
    if max_size is not None:
        scale = max_size / max(w, h)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
    print(f"视频: {total_frames}帧 {fps:.1f}fps, {w}x{h}" + 
          (f" -> {new_w}x{new_h}" if new_w != w else ""))

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
                print(f"抽到上限 {max_frames} 帧")
                break

        frame_idx += 1

    cap.release()
    print(f"抽了 {saved_count} 帧 -> {output_dir}")
    return saved_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/images")
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--max_size", type=int, default=None)
    args = parser.parse_args()

    extract_frames(args.video, args.output_dir, args.interval, args.max_frames, args.max_size)
