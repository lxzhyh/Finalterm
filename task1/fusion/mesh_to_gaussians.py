"""Convert a textured mesh into Graphdeco-style Gaussian PLY initialization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from gaussian_utils import create_gaussians, write_vertices


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = []
        for geom in loaded.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)
        if not meshes:
            raise ValueError(f"No mesh geometry found in {path}")
        return trimesh.util.concatenate(meshes)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    raise TypeError(f"Unsupported mesh type: {type(loaded)!r}")


def texture_image(mesh: trimesh.Trimesh) -> np.ndarray | None:
    visual = mesh.visual
    material = getattr(visual, "material", None)
    image = getattr(material, "image", None)
    if image is None:
        return None
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def barycentric_for_points(triangles: np.ndarray, points: np.ndarray) -> np.ndarray:
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    v0 = b - a
    v1 = c - a
    v2 = points - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = (d00 * d11 - d01 * d01).clip(min=1e-12)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.column_stack([u, v, w])


def sample_texture_colors(mesh: trimesh.Trimesh, points: np.ndarray, face_idx: np.ndarray) -> np.ndarray | None:
    image = texture_image(mesh)
    uv = getattr(mesh.visual, "uv", None)
    if image is None or uv is None:
        return None

    face_uv = uv[mesh.faces[face_idx]]
    triangles = mesh.vertices[mesh.faces[face_idx]]
    bary = barycentric_for_points(triangles, points)
    sample_uv = np.einsum("ij,ijk->ik", bary, face_uv)
    sample_uv = np.mod(sample_uv, 1.0)

    h, w = image.shape[:2]
    px = np.clip((sample_uv[:, 0] * (w - 1)).round().astype(np.int64), 0, w - 1)
    py = np.clip(((1.0 - sample_uv[:, 1]) * (h - 1)).round().astype(np.int64), 0, h - 1)
    return image[py, px, :3]


def sample_visual_colors(mesh: trimesh.Trimesh, face_idx: np.ndarray) -> np.ndarray:
    colors = getattr(mesh.visual, "face_colors", None)
    if colors is not None and len(colors) == len(mesh.faces):
        return np.asarray(colors[face_idx, :3], dtype=np.float32) / 255.0
    colors = getattr(mesh.visual, "vertex_colors", None)
    if colors is not None and len(colors) == len(mesh.vertices):
        face_colors = colors[mesh.faces[face_idx], :3].mean(axis=1)
        return np.asarray(face_colors, dtype=np.float32) / 255.0
    return np.full((len(face_idx), 3), 0.75, dtype=np.float32)


def normalize_points(points: np.ndarray) -> np.ndarray:
    center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    points = points - center
    extent = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    if extent > 1e-8:
        points = points / extent
    return points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--gaussian_scale", type=float, default=0.01)
    parser.add_argument("--opacity", type=float, default=0.8)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--normalize", action="store_true", help="Center mesh and fit it into a unit-scale box.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    mesh = load_mesh(args.mesh)
    points, face_idx = trimesh.sample.sample_surface(mesh, args.samples)
    colors = sample_texture_colors(mesh, points, face_idx)
    if colors is None:
        colors = sample_visual_colors(mesh, face_idx)

    if args.normalize:
        points = normalize_points(points)

    vertices = create_gaussians(
        points,
        colors,
        opacity=args.opacity,
        scale=args.gaussian_scale,
        sh_degree=args.sh_degree,
    )
    write_vertices(args.output, vertices)
    print(f"Wrote {len(vertices)} gaussians: {args.output}")


if __name__ == "__main__":
    main()
