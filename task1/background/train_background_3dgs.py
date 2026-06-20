"""Train a 3DGS background scene from an existing COLMAP dataset."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def find_3dgs_repo() -> Path | None:
    candidates = [
        os.environ.get("GS_REPO_PATH", ""),
        Path(__file__).resolve().parents[1] / "third_party" / "gaussian-splatting",
        Path.home() / "gaussian-splatting",
        Path.home() / "3dgs",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if (path / "train.py").exists():
            return path
    return None


def validate_colmap_scene(scene_path: Path, images: str) -> None:
    image_dir = scene_path / images
    sparse_dir = scene_path / "sparse" / "0"
    missing = []
    if not image_dir.is_dir():
        missing.append(str(image_dir))
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (sparse_dir / name).is_file():
            missing.append(str(sparse_dir / name))
    if missing:
        raise FileNotFoundError("COLMAP scene is incomplete:\n  " + "\n  ".join(missing))


def run_train(
    scene_path: Path,
    output_path: Path,
    images: str,
    iterations: int,
    resolution: int,
    sh_degree: int,
    gpu_id: int,
    render: bool,
) -> None:
    gs_repo = find_3dgs_repo()
    if gs_repo is None:
        raise RuntimeError(
            "Cannot find Graphdeco gaussian-splatting. Set GS_REPO_PATH or clone it "
            "under task1/third_party/gaussian-splatting."
        )

    validate_colmap_scene(scene_path, images)
    output_path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    train_cmd = [
        sys.executable,
        str(gs_repo / "train.py"),
        "-s",
        str(scene_path.resolve()),
        "-m",
        str(output_path.resolve()),
        "-i",
        images,
        "--iterations",
        str(iterations),
        "--test_iterations",
        "7000",
        "15000",
        str(iterations),
        "--save_iterations",
        "7000",
        "15000",
        str(iterations),
        "--sh_degree",
        str(sh_degree),
    ]
    if resolution > 0:
        train_cmd.extend(["--resolution", str(resolution)])

    print("3DGS repo:", gs_repo)
    print("scene:", scene_path)
    print("images:", images)
    print("output:", output_path)
    print("command:", " ".join(train_cmd))

    start = time.time()
    subprocess.run(train_cmd, cwd=gs_repo, env=env, check=True)
    print(f"Training finished in {(time.time() - start) / 60:.1f} min")

    if render:
        render_cmd = [
            sys.executable,
            str(gs_repo / "render.py"),
            "-m",
            str(output_path.resolve()),
        ]
        print("render command:", " ".join(render_cmd))
        subprocess.run(render_cmd, cwd=gs_repo, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene_path",
        type=Path,
        default=Path("data/garden"),
        help="COLMAP scene directory containing images*/ and sparse/0.",
    )
    parser.add_argument(
        "--images",
        default="images_4",
        help="Image directory inside scene_path. Use images_4 first for faster training.",
    )
    parser.add_argument("--output_path", type=Path, default=Path("output/garden"))
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_train(
        scene_path=args.scene_path,
        output_path=args.output_path,
        images=args.images,
        iterations=args.iterations,
        resolution=args.resolution,
        sh_degree=args.sh_degree,
        gpu_id=args.gpu_id,
        render=args.render,
    )


if __name__ == "__main__":
    main()
