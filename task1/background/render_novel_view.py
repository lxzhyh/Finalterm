"""Render a novel-view flythrough from a trained 3DGS background model."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm


def find_3dgs_repo() -> Path:
    candidates = [
        os.environ.get("GS_REPO_PATH", ""),
        Path(__file__).resolve().parents[1] / "third_party" / "gaussian-splatting",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if (path / "gaussian_renderer" / "__init__.py").exists():
            return path
    raise RuntimeError("Cannot find gaussian-splatting. Set GS_REPO_PATH.")


GS_REPO = find_3dgs_repo()
sys.path.insert(0, str(GS_REPO))

from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene.cameras import MiniCam  # noqa: E402
from utils.graphics_utils import focal2fov, getProjectionMatrix  # noqa: E402


def rotation_to_quaternion(rot: np.ndarray) -> np.ndarray:
    trace = np.trace(rot)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (rot[2, 1] - rot[1, 2]) / s,
                (rot[0, 2] - rot[2, 0]) / s,
                (rot[1, 0] - rot[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        axis = int(np.argmax(np.diag(rot)))
        if axis == 0:
            s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            quat = np.array(
                [
                    (rot[2, 1] - rot[1, 2]) / s,
                    0.25 * s,
                    (rot[0, 1] + rot[1, 0]) / s,
                    (rot[0, 2] + rot[2, 0]) / s,
                ],
                dtype=np.float64,
            )
        elif axis == 1:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            quat = np.array(
                [
                    (rot[0, 2] - rot[2, 0]) / s,
                    (rot[0, 1] + rot[1, 0]) / s,
                    0.25 * s,
                    (rot[1, 2] + rot[2, 1]) / s,
                ],
                dtype=np.float64,
            )
        else:
            s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            quat = np.array(
                [
                    (rot[1, 0] - rot[0, 1]) / s,
                    (rot[0, 2] + rot[2, 0]) / s,
                    (rot[1, 2] + rot[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=np.float64,
            )
    return quat / np.linalg.norm(quat)


def quaternion_to_rotation(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        quat = q0 + alpha * (q1 - q0)
        return quat / np.linalg.norm(quat)
    theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    return (
        math.cos(theta) * q0
        + sin_theta * (q1 - dot * q0) / sin_theta_0
    )


def load_camera_path(cameras_json: Path, frames: int) -> tuple[list[np.ndarray], dict[str, float]]:
    cameras = json.loads(cameras_json.read_text(encoding="utf-8"))
    cameras = sorted(cameras, key=lambda item: item["id"])
    positions = [np.asarray(cam["position"], dtype=np.float64) for cam in cameras]
    rotations = [np.asarray(cam["rotation"], dtype=np.float64) for cam in cameras]
    quats = [rotation_to_quaternion(rot) for rot in rotations]

    mats = []
    for frame in range(frames):
        t = frame * len(cameras) / frames
        i0 = int(math.floor(t)) % len(cameras)
        i1 = (i0 + 1) % len(cameras)
        alpha = t - math.floor(t)
        pos = (1.0 - alpha) * positions[i0] + alpha * positions[i1]
        rot = quaternion_to_rotation(slerp(quats[i0], quats[i1], alpha))

        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = rot
        c2w[:3, 3] = pos
        mats.append(c2w)

    ref = cameras[0]
    params = {
        "source_width": ref["width"],
        "source_height": ref["height"],
        "fovx": focal2fov(ref["fx"], ref["width"]),
        "fovy": focal2fov(ref["fy"], ref["height"]),
    }
    return mats, params


def make_camera(c2w: np.ndarray, width: int, height: int, fovx: float, fovy: float) -> MiniCam:
    w2c = np.linalg.inv(c2w).astype(np.float32)
    world_view = torch.tensor(w2c).transpose(0, 1).cuda()
    projection = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    full_proj = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    return MiniCam(width, height, fovy, fovx, 0.01, 100.0, world_view, full_proj)


def tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    image = image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (image * 255.0 + 0.5).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=Path("output/garden"))
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--output_dir", type=Path, default=Path("output/garden/novel_view/ours_30000"))
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--sh_degree", type=int, default=3)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)

    cameras_json = args.model_path / "cameras.json"
    c2w_mats, cam_params = load_camera_path(cameras_json, args.frames)
    height = args.height
    if height <= 0:
        height = round(args.width * cam_params["source_height"] / cam_params["source_width"])
    if height % 2:
        height += 1

    ply = args.model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(str(ply), False)
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / "flythrough.mp4"
    trajectory_path = args.output_dir / "trajectory.json"

    with torch.no_grad(), imageio.get_writer(video_path, fps=args.fps, quality=8, macro_block_size=1) as writer:
        for idx, c2w in enumerate(tqdm(c2w_mats, desc="Novel-view rendering")):
            camera = make_camera(c2w, args.width, height, cam_params["fovx"], cam_params["fovy"])
            image = tensor_to_uint8(render(camera, gaussians, pipeline, background)["render"])
            imageio.imwrite(frames_dir / f"{idx:05d}.png", image)
            writer.append_data(image)

    payload = {
        "description": "Interpolated closed path between COLMAP training cameras.",
        "model_path": str(args.model_path),
        "iteration": args.iteration,
        "frames": args.frames,
        "width": args.width,
        "height": height,
        "fps": args.fps,
        "fovx": cam_params["fovx"],
        "fovy": cam_params["fovy"],
        "camera_to_world": [mat.tolist() for mat in c2w_mats],
    }
    trajectory_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote frames: {frames_dir}")
    print(f"Wrote video: {video_path}")
    print(f"Wrote trajectory: {trajectory_path}")


if __name__ == "__main__":
    main()
