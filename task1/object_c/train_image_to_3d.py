"""单图到 3D，用 Zero123 + threestudio。"""

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path


def find_threestudio_repo():
    candidates = [
        os.environ.get("THREESTUDIO_REPO_PATH", ""),
        os.path.join(os.path.dirname(__file__), "..", "third_party", "threestudio"),
        os.path.expanduser("~/threestudio"),
    ]
    for p in candidates:
        if p and os.path.exists(os.path.join(p, "threestudio")):
            return os.path.abspath(p)
    return None


def train_image_to_3d(config_path: str, image_path: str = None, output_dir: str = "output",
                      max_steps: int = 5000, gpu_id: int = 0):
    os.makedirs(output_dir, exist_ok=True)

    threestudio_repo = find_threestudio_repo()

    if threestudio_repo is not None:
        print(f"threestudio: {threestudio_repo}")

        cmd = [
            sys.executable, "launch.py",
            "--config", os.path.abspath(config_path),
            "--train",
            "--gpu", str(gpu_id),
            f"exp_root_dir={output_dir}",
            f"trainer.max_steps={max_steps}",
        ]

        if image_path:
            cmd.append(f"data.image_path={os.path.abspath(image_path)}")

        print(f"开始 image-to-3d...")
        start_time = time.time()
        result = subprocess.run(cmd, cwd=threestudio_repo)
        elapsed = time.time() - start_time

        if result.returncode != 0:
            raise RuntimeError("threestudio 训练挂了")

        print(f"完成，{elapsed/60:.1f} 分钟")
    else:
        print("没找到 threestudio 仓库，用内置简化版本跑")
        train_image_to_3d_standalone(config_path, image_path, output_dir, max_steps)


def train_image_to_3d_standalone(config_path: str, image_path: str = None,
                                  output_dir: str = "output", max_steps: int = 5000):
    try:
        import torch
        import threestudio
        from omegaconf import OmegaConf
        import pytorch_lightning as pl
    except ImportError:
        print("缺 threestudio 模块，装一下:")
        print("  git clone https://github.com/threestudio-project/threestudio && cd threestudio && pip install -r requirements.txt && pip install -e .")
        sys.exit(1)

    # 加载配置
    cfg = OmegaConf.load(config_path)

    # 覆盖图像路径
    if image_path:
        cfg.data.image_path = os.path.abspath(image_path)
        cfg.system.guidance.cond_image_path = os.path.abspath(image_path)

    # 覆盖输出目录
    cfg.exp_root_dir = output_dir
    cfg.trainer.max_steps = max_steps

    print(f"config: {config_path} | image: {cfg.data.image_path} | output: {output_dir} | max_steps: {max_steps}")

    # 设置随机种子
    pl.seed_everything(cfg.seed, workers=True)

    # 创建数据模块
    dm = threestudio.find(cfg.data_type)(cfg.data)

    # 创建系统
    system = threestudio.find(cfg.system_type)(cfg.system)

    # 创建 trainer
    trainer = pl.Trainer(
        max_steps=cfg.trainer.max_steps,
        accelerator="gpu",
        devices=[0],
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        default_root_dir=output_dir,
    )

    start_time = time.time()

    # 训练
    trainer.fit(system, datamodule=dm)

    elapsed = time.time() - start_time
    print(f"完成，{elapsed/60:.1f} 分钟，模型在 {output_dir}")


def export_mesh(model_path: str, output_path: str, resolution: int = 256):
    print(f"\n导出 mesh: {model_path} -> {output_path} (res {resolution})")
    print(f"用 threestudio 导出: python launch.py --config {model_path}/config.yaml --export --gpu 0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/image-to-3d.yaml")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--export_mesh", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    train_image_to_3d(
        config_path=args.config,
        image_path=args.image,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        gpu_id=args.gpu_id,
    )

    if args.export_mesh:
        export_mesh(args.output_dir, os.path.join(args.output_dir, "mesh"))
