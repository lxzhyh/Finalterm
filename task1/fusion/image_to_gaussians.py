"""Create a lightweight Gaussian asset from an RGBA or plain RGB image.

This fallback is useful when an AIGC asset is available only as a rendered
preview image. It samples foreground pixels into a thin 3D billboard/volume so
the object can still participate in the Gaussian merge-and-render pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from gaussian_utils import create_gaussians, write_vertices


def foreground_mask(image: np.ndarray, alpha_threshold: int, bg_tolerance: float) -> np.ndarray:
    rgb = image[..., :3].astype(np.float32)
    if image.shape[-1] == 4 and image[..., 3].min() < 250:
        return image[..., 3] > alpha_threshold

    corners = np.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]], axis=0)
    bg = np.median(corners, axis=0)
    dist = np.linalg.norm(rgb - bg[None, None, :], axis=-1)
    return dist > bg_tolerance


def sample_image_points(
    image: np.ndarray,
    samples: int,
    width_world: float,
    depth: float,
    alpha_threshold: int,
    bg_tolerance: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = foreground_mask(image, alpha_threshold, bg_tolerance)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("No foreground pixels found.")

    rng = np.random.default_rng(seed)
    replace = len(xs) < samples
    choice = rng.choice(len(xs), size=samples, replace=replace)
    xs = xs[choice]
    ys = ys[choice]

    h, w = image.shape[:2]
    aspect = h / max(w, 1)
    height_world = width_world * aspect

    x = (xs / max(w - 1, 1) - 0.5) * width_world
    y = (0.5 - ys / max(h - 1, 1)) * height_world
    z = rng.uniform(-depth * 0.5, depth * 0.5, size=samples)
    xyz = np.column_stack([x, y, z]).astype(np.float32)
    rgb = image[ys, xs, :3].astype(np.float32) / 255.0
    return xyz, rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--width_world", type=float, default=1.0)
    parser.add_argument("--depth", type=float, default=0.04)
    parser.add_argument("--opacity", type=float, default=0.8)
    parser.add_argument("--gaussian_scale", type=float, default=0.015)
    parser.add_argument("--alpha_threshold", type=int, default=8)
    parser.add_argument("--bg_tolerance", type=float, default=35.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = np.asarray(Image.open(args.image).convert("RGBA"))
    xyz, rgb = sample_image_points(
        image,
        samples=args.samples,
        width_world=args.width_world,
        depth=args.depth,
        alpha_threshold=args.alpha_threshold,
        bg_tolerance=args.bg_tolerance,
        seed=args.seed,
    )
    vertices = create_gaussians(
        xyz,
        rgb,
        opacity=args.opacity,
        scale=args.gaussian_scale,
        sh_degree=3,
    )
    write_vertices(args.output, vertices)
    print(f"Wrote {len(vertices)} gaussians: {args.output}")


if __name__ == "__main__":
    main()
