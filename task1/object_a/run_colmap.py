"""
调用 COLMAP 进行特征提取、匹配和稀疏重建 (SfM)。
需要系统已安装 COLMAP 命令行工具。

用法:
    python run_colmap.py --image_dir data/images --workspace data
"""

import argparse
import os
import subprocess
import shutil
from pathlib import Path


def run_command(cmd: list, desc: str):
    """执行外部命令并打印结果（输出实时流到终端）"""
    print(f"\n{'='*60}")
    print(f"[COLMAP] {desc}")
    print(f"{'='*60}")
    print(f"命令: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"COLMAP {desc} 失败 (exit code: {result.returncode})")
    print(f"[完成] {desc}")
    return result


def run_colmap_pipeline(image_dir: str, workspace: str, camera_model: str = "OPENCV",
                        single_camera: bool = True, gpu_id: str = "0"):
    """
    完整的 COLMAP SfM 流程 (SIFT 使用 CPU 模式，兼容无头服务器):
    1. feature_extractor  - 提取 SIFT 特征
    2. exhaustive_matcher - 穷举匹配
    3. mapper             - 稀疏重建
    4. undistortion       - 去畸变 (3DGS 需要)
    """
    database_path = os.path.join(workspace, "database.db")
    sparse_dir = os.path.join(workspace, "sparse")
    dense_dir = os.path.join(workspace, "dense")

    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(dense_dir, exist_ok=True)

    # 如果已有旧的 database，先删除
    if os.path.exists(database_path):
        os.remove(database_path)

    # Step 1: 特征提取 (CPU 模式，避免无头服务器 OpenGL 问题)
    # 增加特征点数以提升视频序列注册率
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
    run_command(feat_cmd, "特征提取 (Feature Extraction)")

    # Step 2: 顺序匹配 (视频序列更适合 sequential matcher)
    # 每帧与邻近 20 帧进行匹配，覆盖足够视差同时避免无效匹配
    match_cmd = [
        "colmap", "sequential_matcher",
        "--database_path", database_path,
        "--SiftMatching.use_gpu", "0",
        "--SequentialMatching.overlap", "20",
    ]
    run_command(match_cmd, "穷举匹配 (Exhaustive Matching)")

    # Step 3: 稀疏重建 (mapper)
    map_cmd = [
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--output_path", sparse_dir,
    ]
    run_command(map_cmd, "稀疏重建 (Sparse Reconstruction)")

    # 检查重建结果
    recon_dirs = sorted(Path(sparse_dir).glob("*"))
    if not recon_dirs:
        raise RuntimeError("COLMAP 重建失败，未生成任何稀疏模型")

    best_recon = recon_dirs[0]
    print(f"\n重建成功，使用模型: {best_recon}")

    # Step 4: 去畸变 (为 3DGS 准备)
    undist_cmd = [
        "colmap", "image_undistorter",
        "--image_path", image_dir,
        "--input_path", str(best_recon),
        "--output_path", dense_dir,
        "--output_type", "COLMAP",
    ]
    run_command(undist_cmd, "图像去畸变 (Undistortion)")

    # 3DGS 期望 sparse/0/ 子目录结构，而 image_undistorter 将文件直接放在 sparse/ 下
    dense_sparse = os.path.join(dense_dir, "sparse")
    sparse_0 = os.path.join(dense_sparse, "0")
    if not os.path.exists(sparse_0):
        os.makedirs(sparse_0, exist_ok=True)
        for fname in os.listdir(dense_sparse):
            src = os.path.join(dense_sparse, fname)
            dst = os.path.join(sparse_0, fname)
            if os.path.isfile(src):
                shutil.move(src, dst)
        print(f"[重组] 稀疏模型已移动到: {sparse_0}")

    print(f"\n{'='*60}")
    print(f"COLMAP 流程完成!")
    print(f"稀疏模型: {best_recon}")
    print(f"去畸变数据: {dense_dir}")
    print(f"{'='*60}")

    return dense_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 COLMAP SfM 流程")
    parser.add_argument("--image_dir", type=str, default="data/images", help="输入图像目录")
    parser.add_argument("--workspace", type=str, default="data", help="COLMAP 工作目录")
    parser.add_argument("--camera_model", type=str, default="OPENCV",
                        choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"],
                        help="相机模型")
    parser.add_argument("--single_camera", action="store_true", default=True,
                        help="所有图像使用同一相机内参")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU 设备 ID")
    args = parser.parse_args()

    run_colmap_pipeline(
        image_dir=args.image_dir,
        workspace=args.workspace,
        camera_model=args.camera_model,
        single_camera=args.single_camera,
        gpu_id=args.gpu_id,
    )
