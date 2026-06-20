"""Render a lightweight flythrough preview from a Gaussian PLY file.

This is a fast layout/debug renderer. It splats Gaussian centers as colored
points and is intended to verify scale, placement and camera path before using
the full Graphdeco renderer for final-quality frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from gaussian_utils import extract_rgb, read_vertices


def look_at_camera(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward).clip(min=1e-8)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right).clip(min=1e-8)
    true_up = np.cross(right, forward)

    world_to_cam = np.eye(4, dtype=np.float32)
    world_to_cam[:3, :3] = np.stack([right, true_up, -forward], axis=0)
    world_to_cam[:3, 3] = -world_to_cam[:3, :3] @ eye
    return world_to_cam


def render_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    world_to_cam: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    point_radius: int,
) -> np.ndarray:
    cam = (world_to_cam[:3, :3] @ xyz.T + world_to_cam[:3, 3:4]).T
    z = -cam[:, 2]
    valid = z > 1e-4
    cam = cam[valid]
    z = z[valid]
    colors = rgb[valid]

    focal = 0.5 * width / np.tan(np.deg2rad(fov_deg) * 0.5)
    u = (focal * cam[:, 0] / z + width * 0.5).round().astype(np.int32)
    v = (height * 0.5 - focal * cam[:, 1] / z).round().astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z, colors = u[inside], v[inside], z[inside], colors[inside]

    order = np.argsort(z)[::-1]
    image = np.zeros((height, width, 3), dtype=np.float32)
    radius = max(point_radius, 1)
    for idx in order:
        x0 = max(0, u[idx] - radius)
        x1 = min(width, u[idx] + radius + 1)
        y0 = max(0, v[idx] - radius)
        y1 = min(height, v[idx] + radius + 1)
        image[y0:y1, x0:x1] = colors[idx]
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def trajectory(center: np.ndarray, radius: float, height: float, frames: int) -> list[np.ndarray]:
    mats = []
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    for i in range(frames):
        theta = 2.0 * np.pi * i / frames
        eye = center + np.array([radius * np.cos(theta), radius * np.sin(theta), height])
        mats.append(look_at_camera(eye, center, up))
    return mats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/fused_scene/flythrough_preview.mp4"))
    parser.add_argument("--trajectory_json", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument("--camera_height", type=float, default=1.2)
    parser.add_argument("--center", type=float, nargs=3, default=None)
    parser.add_argument("--point_radius", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth gaussian for faster preview.")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    vertices = read_vertices(args.ply)
    xyz = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32)
    rgb = extract_rgb(vertices)
    if args.stride > 1:
        xyz = xyz[:: args.stride]
        rgb = rgb[:: args.stride]

    center = np.asarray(args.center, dtype=np.float32) if args.center else np.median(xyz, axis=0)
    mats = trajectory(center, args.radius, args.camera_height, args.frames)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.output, fps=args.fps, quality=8, macro_block_size=1) as writer:
        for mat in mats:
            writer.append_data(
                render_points(
                    xyz,
                    rgb,
                    mat,
                    args.width,
                    args.height,
                    args.fov,
                    args.point_radius,
                )
            )

    if args.trajectory_json:
        args.trajectory_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "width": args.width,
            "height": args.height,
            "fov_deg": args.fov,
            "world_to_camera": [mat.tolist() for mat in mats],
        }
        args.trajectory_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote preview video: {args.output}")


if __name__ == "__main__":
    main()
