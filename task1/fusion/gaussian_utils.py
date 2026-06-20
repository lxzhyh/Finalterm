"""Utilities for Graphdeco-style 3D Gaussian PLY files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class Transform:
    scale: float = 1.0
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def rotation(self) -> Rotation:
        return Rotation.from_euler("xyz", self.rotation_deg, degrees=True)


def rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max(initial=0.0) > 1.0:
        rgb = rgb / 255.0
    return (rgb - 0.5) / SH_C0


def sh_dc_to_rgb(dc: np.ndarray) -> np.ndarray:
    rgb = np.asarray(dc, dtype=np.float32) * SH_C0 + 0.5
    return np.clip(rgb, 0.0, 1.0)


def read_vertices(path: str | Path) -> np.ndarray:
    ply = PlyData.read(str(path))
    return ply["vertex"].data


def write_vertices(path: str | Path, vertices: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))


def names_with_prefix(dtype: np.dtype, prefix: str) -> list[str]:
    return sorted(
        [name for name in dtype.names or () if name.startswith(prefix)],
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )


def make_gaussian_dtype(sh_degree: int = 3) -> np.dtype:
    f_rest_count = 3 * ((sh_degree + 1) ** 2 - 1)
    fields: list[tuple[str, str]] = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ]
    fields.extend((f"f_rest_{i}", "f4") for i in range(f_rest_count))
    fields.extend(
        [
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
        ]
    )
    return np.dtype(fields)


def create_gaussians(
    xyz: np.ndarray,
    rgb: np.ndarray,
    opacity: float = 0.8,
    scale: float = 0.01,
    sh_degree: int = 3,
) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim == 1:
        rgb = np.repeat(rgb[None, :], len(xyz), axis=0)

    vertices = np.zeros(len(xyz), dtype=make_gaussian_dtype(sh_degree))
    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]
    dc = rgb_to_sh_dc(rgb[:, :3])
    vertices["f_dc_0"] = dc[:, 0]
    vertices["f_dc_1"] = dc[:, 1]
    vertices["f_dc_2"] = dc[:, 2]
    vertices["opacity"] = np.log(opacity / max(1.0 - opacity, 1e-6))
    log_scale = np.log(max(scale, 1e-8))
    vertices["scale_0"] = log_scale
    vertices["scale_1"] = log_scale
    vertices["scale_2"] = log_scale
    vertices["rot_0"] = 1.0
    return vertices


def align_dtype(vertices: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    aligned = np.zeros(len(vertices), dtype=target_dtype)
    source_names = set(vertices.dtype.names or ())
    for name in target_dtype.names or ():
        if name in source_names:
            aligned[name] = vertices[name]
    if "rot_0" in target_dtype.names and "rot_0" not in source_names:
        aligned["rot_0"] = 1.0
    return aligned


def apply_transform(vertices: np.ndarray, transform: Transform) -> np.ndarray:
    out = vertices.copy()
    xyz = np.column_stack([out["x"], out["y"], out["z"]]).astype(np.float32)
    rot = transform.rotation
    xyz = rot.apply(xyz * transform.scale) + np.asarray(transform.translation, dtype=np.float32)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    scale_names = [name for name in ("scale_0", "scale_1", "scale_2") if name in out.dtype.names]
    if scale_names:
        log_s = np.log(max(transform.scale, 1e-8))
        for name in scale_names:
            out[name] = out[name] + log_s

    rot_names = [name for name in ("rot_0", "rot_1", "rot_2", "rot_3") if name in out.dtype.names]
    if len(rot_names) == 4:
        q_old_wxyz = np.column_stack([out[name] for name in rot_names]).astype(np.float64)
        q_old_xyzw = q_old_wxyz[:, [1, 2, 3, 0]]
        q_new_xyzw = (rot * Rotation.from_quat(q_old_xyzw)).as_quat()
        q_new_wxyz = q_new_xyzw[:, [3, 0, 1, 2]]
        q_new_wxyz /= np.linalg.norm(q_new_wxyz, axis=1, keepdims=True).clip(min=1e-8)
        for i, name in enumerate(rot_names):
            out[name] = q_new_wxyz[:, i]

    return out


def extract_rgb(vertices: np.ndarray) -> np.ndarray:
    names = vertices.dtype.names or ()
    if all(name in names for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        dc = np.column_stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]])
        return sh_dc_to_rgb(dc)
    if all(name in names for name in ("red", "green", "blue")):
        return np.column_stack([vertices["red"], vertices["green"], vertices["blue"]]) / 255.0
    return np.full((len(vertices), 3), 0.8, dtype=np.float32)
