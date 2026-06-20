"""Merge background and object Gaussian PLY files into one 3DGS scene."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from gaussian_utils import Transform, align_dtype, apply_transform, read_vertices, write_vertices


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_transform(cfg: dict) -> Transform:
    return Transform(
        scale=float(cfg.get("scale", 1.0)),
        rotation_deg=tuple(float(v) for v in cfg.get("rotation_deg", [0.0, 0.0, 0.0])),
        translation=tuple(float(v) for v in cfg.get("translation", [0.0, 0.0, 0.0])),
    )


def iteration_dir(output_model: Path, iteration: int) -> Path:
    return output_model / "point_cloud" / f"iteration_{iteration}"


def copy_optional_model_files(background_model: Path, output_model: Path) -> None:
    for name in ("cfg_args", "cameras.json", "input.ply"):
        src = background_model / name
        if src.exists():
            shutil.copy2(src, output_model / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/scene_layout.yaml"))
    parser.add_argument("--output_model", type=Path, default=Path("output/fused_scene"))
    parser.add_argument("--iteration", type=int, default=30000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    background_path = Path(cfg["background"]["ply"])
    background_model = Path(cfg["background"].get("model_dir", background_path.parents[2]))

    merged_parts = []
    base = read_vertices(background_path)
    target_dtype = base.dtype
    merged_parts.append(apply_transform(base, parse_transform(cfg.get("background", {}))))

    for item in cfg.get("objects", []):
        if not item.get("enabled", True):
            continue
        name = item.get("name", Path(item["ply"]).stem)
        vertices = read_vertices(item["ply"])
        vertices = align_dtype(vertices, target_dtype)
        vertices = apply_transform(vertices, parse_transform(item))
        merged_parts.append(vertices)
        print(f"Added {name}: {len(vertices)} gaussians")

    merged = np.concatenate(merged_parts)
    out_ply = iteration_dir(args.output_model, args.iteration) / "point_cloud.ply"
    write_vertices(out_ply, merged)
    copy_optional_model_files(background_model, args.output_model)

    metadata = {
        "config": str(args.config),
        "background": str(background_path),
        "iteration": args.iteration,
        "num_gaussians": int(len(merged)),
        "parts": [
            {"name": "background", "path": str(background_path), "count": int(len(base))}
        ]
        + [
            {
                "name": item.get("name", Path(item["ply"]).stem),
                "path": str(item["ply"]),
                "enabled": bool(item.get("enabled", True)),
            }
            for item in cfg.get("objects", [])
        ],
    }
    args.output_model.mkdir(parents=True, exist_ok=True)
    (args.output_model / "fusion_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Wrote merged scene: {out_ply}")
    print(f"Total gaussians: {len(merged)}")


if __name__ == "__main__":
    main()
