"""跑 COLMAP SfM 流程：提特征 → 匹配 → 稀疏重建 → 去畸变。"""

import argparse
import os
import subprocess
import shutil
from pathlib import Path


def run_command(cmd: list, desc: str):
    print(f"\n--- COLMAP: {desc} ---")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"{desc} 挂了 (exit {result.returncode})")
    return result


def run_colmap_pipeline(image_dir: str, workspace: str, camera_model: str = "OPENCV",
                        single_camera: bool = True, gpu_id: str = "0"):
    database_path = os.path.join(workspace, "database.db")
    sparse_dir = os.path.join(workspace, "sparse")
    dense_dir = os.path.join(workspace, "dense")

    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(dense_dir, exist_ok=True)

    if os.path.exists(database_path):
        os.remove(database_path)

    # 特征提取，CPU 模式避免无头服务器 OpenGL 问题
    feat_cmd = [
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1" if single_camera else "0",
        "--SiftExtraction.use_gpu", "0",
        "--SiftExtraction.max_num_features", "8192",
        "--SiftExtraction.first_octave", "-1",
    ]
    run_command(feat_cmd, "特征提取")

    # 视频序列用 sequential matcher，邻帧 20 张
    match_cmd = [
        "colmap", "sequential_matcher",
        "--database_path", database_path,
        "--SiftMatching.use_gpu", "0",
        "--SequentialMatching.overlap", "20",
    ]
    run_command(match_cmd, "特征匹配")

    map_cmd = [
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--output_path", sparse_dir,
    ]
    run_command(map_cmd, "稀疏重建")

    recon_dirs = sorted(Path(sparse_dir).glob("*"))
    if not recon_dirs:
        raise RuntimeError("没跑出稀疏模型，挂了")

    best_recon = recon_dirs[0]
    print(f"  重建模型: {best_recon}")

    # 去畸变，3DGS 需要
    undist_cmd = [
        "colmap", "image_undistorter",
        "--image_path", image_dir,
        "--input_path", str(best_recon),
        "--output_path", dense_dir,
        "--output_type", "COLMAP",
    ]
    run_command(undist_cmd, "去畸变")

    # 3DGS 要 sparse/0/ 结构，image_undistorter 直接放 sparse/ 下了
    dense_sparse = os.path.join(dense_dir, "sparse")
    sparse_0 = os.path.join(dense_sparse, "0")
    if not os.path.exists(sparse_0):
        os.makedirs(sparse_0, exist_ok=True)
        for fname in os.listdir(dense_sparse):
            src = os.path.join(dense_sparse, fname)
            dst = os.path.join(sparse_0, fname)
            if os.path.isfile(src):
                shutil.move(src, dst)

    print(f"\n搞定，稀疏模型: {best_recon}, 去畸变数据: {dense_dir}")

    return dense_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="data/images")
    parser.add_argument("--workspace", type=str, default="data")
    parser.add_argument("--camera_model", type=str, default="OPENCV",
                        choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"])
    parser.add_argument("--single_camera", action="store_true", default=True)
    parser.add_argument("--gpu_id", type=str, default="0")
    args = parser.parse_args()

    run_colmap_pipeline(
        image_dir=args.image_dir,
        workspace=args.workspace,
        camera_model=args.camera_model,
        single_camera=args.single_camera,
        gpu_id=args.gpu_id,
    )
