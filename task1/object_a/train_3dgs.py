"""训 3D Gaussian Splatting，需要 COLMAP 结果作为输入。"""

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path


def find_3dgs_repo():
    candidates = [
        os.environ.get("GS_REPO_PATH", ""),
        os.path.join(os.path.dirname(__file__), "..", "third_party", "gaussian-splatting"),
        os.path.expanduser("~/gaussian-splatting"),
        os.path.expanduser("~/3dgs"),
    ]
    for p in candidates:
        if p and os.path.exists(os.path.join(p, "train.py")):
            return os.path.abspath(p)
    return None


def train_3dgs(source_path: str, output_path: str, iterations: int = 30000,
               test_iterations: list = None, save_iterations: list = None,
               resolution: int = -1, sh_degree: int = 3, gpu_id: int = 0):
    if test_iterations is None:
        test_iterations = [7000, 15000, iterations]
    if save_iterations is None:
        save_iterations = [7000, 15000, iterations]

    os.makedirs(output_path, exist_ok=True)

    gs_repo = find_3dgs_repo()

    if gs_repo is not None:
        print(f"3DGS 仓库: {gs_repo}")

        source_path = os.path.abspath(source_path)
        output_path = os.path.abspath(output_path)

        cmd = [
            sys.executable, os.path.join(gs_repo, "train.py"),
            "-s", source_path,
            "-m", output_path,
            "--iterations", str(iterations),
            "--test_iterations", *[str(i) for i in test_iterations],
            "--save_iterations", *[str(i) for i in save_iterations],
            "--sh_degree", str(sh_degree),
        ]
        if resolution > 0:
            cmd.extend(["--resolution", str(resolution)])

        print("开始训 3DGS...")
        start_time = time.time()
        result = subprocess.run(cmd, cwd=gs_repo)
        elapsed = time.time() - start_time

        if result.returncode != 0:
            raise RuntimeError("3DGS 训练挂了")

        print(f"训完，耗时 {elapsed/60:.1f} 分钟")
    else:
        print("没找到 3DGS 仓库，用内置简化版本跑")
        train_3dgs_standalone(source_path, output_path, iterations, resolution, sh_degree)


def train_3dgs_standalone(source_path: str, output_path: str, iterations: int = 30000,
                           resolution: int = -1, sh_degree: int = 3):
    try:
        import torch
        from gaussian_splatting.scene import Scene
        from gaussian_splatting.gaussian_model import GaussianModel
        from gaussian_splatting.train import training_setup, train_iteration
        from gaussian_splatting.utils import safe_state
    except ImportError:
        print("缺 gaussian_splatting 模块，装一下:")
        print("  git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive")
        print("  cd gaussian-splatting && pip install submodules/diff-gaussian-rasterization submodules/simple-knn && pip install -e .")
        sys.exit(1)

    from argparse import Namespace

    args = Namespace(
        source_path=source_path,
        model_path=output_path,
        images="images",
        resolution=resolution,
        white_background=False,
        data_device="cuda",
        eval=False,
        iterations=iterations,
        test_iterations=[7000, 15000, iterations],
        save_iterations=[7000, 15000, iterations],
        checkpoint_iterations=[],
        checkpoint=None,
        sh_degree=sh_degree,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30000,
        feature_lr=0.0025,
        opacity_lr=0.05,
        scaling_lr=0.005,
        rotation_lr=0.001,
        percent_dense=0.01,
        lambda_dssim=0.2,
        densification_interval=100,
        opacity_reset_interval=3000,
        densify_from_iter=500,
        densify_until_iter=15000,
        densify_grad_threshold=0.0002,
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )

    safe_state(False)

    print(f"数据: {source_path} | 输出: {output_path} | iter: {iterations} | SH: {sh_degree}")

    start_time = time.time()

    dataset_params = {
        "source_path": args.source_path,
        "images": args.images,
        "resolution": args.resolution,
        "white_background": args.white_background,
        "data_device": args.data_device,
        "eval": args.eval,
    }

    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(dataset_params, gaussians)
    gaussians.training_setup(args)

    first_iter = 0
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    progress_bar = None
    try:
        from tqdm import tqdm
        progress_bar = tqdm(range(first_iter, args.iterations), desc="Training")
    except ImportError:
        pass

    for iteration in range(first_iter + 1, args.iterations + 1):
        train_iteration(gaussians, scene, background, args, iteration)

        if progress_bar is not None:
            progress_bar.update(1)

        if iteration in args.save_iterations:
            print(f"\n  saving @ iter {iteration}")
            gaussians.save_ply(os.path.join(args.model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply"))

    elapsed = time.time() - start_time
    print(f"\n训完 {elapsed/60:.1f} 分钟，模型在 {output_path}")


def render_3dgs(model_path: str, output_path: str, skip_train: bool = False, skip_test: bool = False):
    gs_repo = find_3dgs_repo()

    if gs_repo is not None:
        model_path = os.path.abspath(model_path)
        cmd = [
            sys.executable, os.path.join(gs_repo, "render.py"),
            "-m", model_path,
            "--skip_train" if skip_train else "",
            "--skip_test" if skip_test else "",
        ]
        cmd = [c for c in cmd if c]
        subprocess.run(cmd, cwd=gs_repo)
    else:
        print(f"没找到 3DGS 仓库，请用 render.py 或 SIBR_viewer 渲染，模型在: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, default="data/dense")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    train_3dgs(
        source_path=args.source_path,
        output_path=args.output_path,
        iterations=args.iterations,
        resolution=args.resolution,
        sh_degree=args.sh_degree,
        gpu_id=args.gpu_id,
    )

    if args.render:
        render_3dgs(args.output_path, os.path.join(args.output_path, "render"))
