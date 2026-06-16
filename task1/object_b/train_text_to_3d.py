"""
使用 threestudio 框架进行文本到 3D 生成。
基于预训练的 2D 扩散模型 (Stable Diffusion) 与 SDS Loss。

用法:
    python train_text_to_3d.py --config configs/text-to-3d.yaml --prompt "a cute dragon"
"""

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path


def find_threestudio_repo():
    """查找 threestudio 代码仓库位置"""
    candidates = [
        os.environ.get("THREESTUDIO_REPO_PATH", ""),
        os.path.join(os.path.dirname(__file__), "..", "third_party", "threestudio"),
        os.path.expanduser("~/threestudio"),
    ]
    for p in candidates:
        if p and os.path.exists(os.path.join(p, "threestudio")):
            return os.path.abspath(p)
    return None


def train_text_to_3d(config_path: str, prompt: str = None, output_dir: str = "output",
                     max_steps: int = 10000, gpu_id: int = 0):
    """
    调用 threestudio 进行文本到 3D 生成。
    """
    os.makedirs(output_dir, exist_ok=True)

    threestudio_repo = find_threestudio_repo()

    if threestudio_repo is not None:
        print(f"找到 threestudio 仓库: {threestudio_repo}")

        cmd = [
            sys.executable, "launch.py",
            "--config", os.path.abspath(config_path),
            "--train",
            "--gpu", str(gpu_id),
            f"exp_root_dir={output_dir}",
            f"trainer.max_steps={max_steps}",
        ]

        if prompt:
            cmd.append(f"system.prompt_processor.prompt={prompt}")

        print(f"\n开始文本到 3D 生成...")
        print(f"命令: {' '.join(cmd)}")

        start_time = time.time()
        result = subprocess.run(cmd, cwd=threestudio_repo)
        elapsed = time.time() - start_time

        if result.returncode != 0:
            raise RuntimeError("threestudio 训练失败")

        print(f"\n生成完成! 耗时: {elapsed/60:.1f} 分钟")
    else:
        print("[警告] 未找到 threestudio 仓库，使用内置简化脚本")
        print("请按照 README 中的说明安装 threestudio 后重新运行")
        train_text_to_3d_standalone(config_path, prompt, output_dir, max_steps)


def train_text_to_3d_standalone(config_path: str, prompt: str = None,
                                 output_dir: str = "output", max_steps: int = 10000):
    """
    独立的文本到 3D 生成入口。
    适用于已将 threestudio 作为 Python 包安装的情况。
    """
    try:
        import torch
        import threestudio
        from omegaconf import OmegaConf
        import pytorch_lightning as pl
        from threestudio.utils.config import parse_structured
        from threestudio.utils.misc import get_rank
    except ImportError:
        print("\n[错误] 无法导入 threestudio 模块。")
        print("请先安装 threestudio:")
        print("  git clone https://github.com/threestudio-project/threestudio")
        print("  cd threestudio")
        print("  pip install -r requirements.txt")
        print("  pip install -e .")
        sys.exit(1)

    # 加载配置
    cfg = OmegaConf.load(config_path)

    # 覆盖 prompt
    if prompt:
        cfg.system.prompt_processor.prompt = prompt

    # 覆盖输出目录
    cfg.exp_root_dir = output_dir
    cfg.trainer.max_steps = max_steps

    print(f"\n{'='*60}")
    print(f"文本到 3D 生成参数:")
    print(f"  配置文件: {config_path}")
    print(f"  Prompt: {cfg.system.prompt_processor.prompt}")
    print(f"  输出目录: {output_dir}")
    print(f"  最大步数: {max_steps}")
    print(f"{'='*60}\n")

    # 设置随机种子
    pl.seed_everything(cfg.seed, workers=True)

    # 创建数据模块
    dm = threestudio.find(cfg.data_type)(cfg.data)

    # 创建系统
    system = threestudio.find(cfg.system_type)(
        cfg.system,
        uncond_processor_pretrained_model_name_or_path=None,
    )

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
    print(f"\n生成完成! 总耗时: {elapsed/60:.1f} 分钟")
    print(f"模型已保存至: {output_dir}")


def export_mesh(model_path: str, output_path: str, resolution: int = 256):
    """
    将隐式场导出为 Mesh (用于后续场景融合)。
    """
    try:
        import torch
        import trimesh
        import mcubes
    except ImportError:
        print("[错误] 导出 Mesh 需要安装 trimesh 和 mcubes")
        return

    print(f"\n导出 Mesh...")
    print(f"  模型路径: {model_path}")
    print(f"  分辨率: {resolution}")
    print(f"  输出路径: {output_path}")

    # 这里需要根据实际的 threestudio 模型格式来实现加载和导出逻辑
    # 以下为示例代码
    print("[提示] 请使用 threestudio 官方导出脚本:")
    print(f"  python launch.py --config {model_path}/config.yaml --export --gpu 0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="文本到 3D 生成 (threestudio)")
    parser.add_argument("--config", type=str, default="configs/text-to-3d.yaml",
                        help="配置文件路径")
    parser.add_argument("--prompt", type=str, default=None,
                        help="文本 Prompt (覆盖配置文件中的 prompt)")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="输出目录")
    parser.add_argument("--max_steps", type=int, default=10000,
                        help="最大训练步数")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU 设备 ID")
    parser.add_argument("--export_mesh", action="store_true",
                        help="训练后导出 Mesh")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    train_text_to_3d(
        config_path=args.config,
        prompt=args.prompt,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        gpu_id=args.gpu_id,
    )

    if args.export_mesh:
        export_mesh(args.output_dir, os.path.join(args.output_dir, "mesh"))
