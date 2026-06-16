# 题目一：3D 资产准备

本项目实现了三种不同的 3D 物体生成方法，用于后续的 3DGS 场景融合。

## 项目结构

```
task1/
├── object_a/              # 物体 A: 真实多视角重建 (COLMAP + 3DGS)
│   ├── extract_frames.py  # 从视频抽帧
│   ├── run_colmap.py      # COLMAP SfM 流程
│   ├── train_3dgs.py      # 3DGS 训练
│   ├── run_pipeline.sh    # 完整流水线脚本
│   └── data/              # 数据目录
│       ├── images/        # 帧图像
│       └── sparse/        # COLMAP 稀疏模型
├── object_b/              # 物体 B: 文本到 3D 生成 (threestudio + SDS)
│   ├── train_text_to_3d.py
│   ├── run_pipeline.sh
│   └── configs/
│       └── text-to-3d.yaml
├── object_c/              # 物体 C: 单图到 3D 生成 (Zero123)
│   ├── remove_background.py
│   ├── train_image_to_3d.py
│   ├── run_pipeline.sh
│   ├── configs/
│   │   └── image-to-3d.yaml
│   └── data/
│       ├── raw/           # 原始图像
│       └── processed/     # 去背景后的图像
├── requirements.txt       # Python 依赖
└── environment.yml        # Conda 环境配置
```

## 环境配置

### 创建环境

```bash
conda env create -f environment.yml
conda activate hw3d_assets
```

或者手动来：

```bash
conda create -n hw3d_assets python=3.10
conda activate hw3d_assets
pip install -r requirements.txt
```

### 装 COLMAP

```bash
sudo apt-get install colmap
```

要 GPU 加速就从源码编译：

```bash
git clone https://github.com/colmap/colmap.git
cd colmap && mkdir build && cd build
cmake .. -DCUDA_ENABLED=ON
make -j$(nproc)
sudo make install
```

conda 方式也行：

```bash
conda install -c conda-forge colmap
conda install faiss openimageio
```

### 装 3D Gaussian Splatting

```bash
cd ../third_party
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting

sed -i '1i #include <cstdint>' submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h
pip install --no-build-isolation submodules/diff-gaussian-rasterization
pip install --no-build-isolation submodules/simple-knn
pip install plyfile tqdm
```

设个环境变量方便找：

```bash
export GS_REPO_PATH=/path/to/gaussian-splatting
```

### 装 threestudio

```bash
cd ~/homework/Finalterm/task1/third_party
git clone https://github.com/threestudio-project/threestudio.git
cd threestudio

git clone https://github.com/NVlabs/nvdiffrast.git
pip install --no-build-isolation -e nvdiffrast

pip install --no-build-isolation -r requirements.txt
pip install pybind11
pip install --no-build-isolation pysdf
CXXFLAGS="-std=c++14" pip install --no-build-isolation nerfacc==0.5.2

pip install --no-build-isolation -e .
```

```bash
export THREESTUDIO_REPO_PATH=/path/to/threestudio
```

### 运行时环境变量

这些已经写进各个 `run_pipeline.sh` 了，一般不用手动设：

```bash
export CUDA_HOME=$CONDA_PREFIX
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH
export HF_ENDPOINT=https://hf-mirror.com
```

### 下载预训练模型

训练脚本首次运行会自动从 hf-mirror 拉，网络不稳的话提前下好：

```bash
# Stable Diffusion 2.1（物体 B 用）
hf download Manojb/stable-diffusion-2-1-base --local-dir ./models/sd-2-1-base

# Zero123-XL（物体 C 用）
hf download bennyguo/zero123-xl-diffusers --local-dir ./models/zero123-xl

# rembg u2net（去背景用，GitHub 容易超时，建议提前下）
wget -O ~/.u2net/u2net.onnx \
  https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx

# SAM（可选，精细去背景）
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P ~/models/
```

## 物体 A: 真实多视角重建

用手机拍一段物体的环绕视频（30-60 秒，保持物体静止、相机平稳绕一圈），然后 COLMAP 算位姿 + 3DGS 重建。

把视频放到 `object_a/data/` 下，然后：

```bash
cd object_a
bash run_pipeline.sh --video data/video.mp4 --gpu 7 --max_frames 100
```

常用参数：`--interval` 抽帧间隔（默认 2），`--max_frames` 最多抽多少帧（默认 300），`--iterations` 3DGS 训练轮数（默认 30000），`--resolution` 图片缩放（-1 表示原尺寸）。

想一步步跑也行：

```bash
python extract_frames.py --video data/video.mp4 --output_dir data/images --interval 2
python run_colmap.py --image_dir data/images --workspace data --gpu_id 0
python train_3dgs.py --source_path data/dense --output_path output --iterations 30000 --gpu_id 7 --render
```

输出：`data/images/`（抽的帧）、`data/sparse/`（COLMAP 稀疏模型）、`data/dense/`（去畸变图+相机参数）、`output/`（3DGS 模型，含 point_cloud.ply）

## 物体 B: 文本到 3D 生成

基于 threestudio + Stable Diffusion 2.1，写一句话描述，SDS 优化出 3D 物体。

```bash
cd object_b
bash run_pipeline.sh --prompt "a 3D model of a wooden chair" --max_steps 10000 --gpu 7
```

参数：`--config` 配置文件（默认 `configs/text-to-3d.yaml`），`--prompt` 文本描述，`--max_steps` 最大训练步数，`--gpu` 用哪张卡。

注意：实际输出在 `third_party/threestudio/output/text-to-3d-sds/<prompt>@<timestamp>/` 下，不在 `object_b/output/`。里面有 `ckpts/`（checkpoint）、`save/`（渲染图）、`csv_logs/`（训练日志）。

导出 Mesh：

```bash
cd third_party/threestudio
python launch.py \
  --config "output/text-to-3d-sds/<你的实验目录>/configs/parsed.yaml" \
  --export --gpu 7 \
  "resume=output/text-to-3d-sds/<你的实验目录>/ckpts/last.ckpt"
```

导出产物在 `<实验目录>/save/it<步数>-export/`：`model.obj` + `model.mtl` + `texture_kd.jpg`。

## 物体 C: 单图到 3D 生成

给一张照片，先去背景，再用 Zero123 + threestudio 推理出 3D。照片放 `object_c/data/raw/`（已有一张 `photo.png`）。

```bash
pip install mediapipe
cd object_c
bash run_pipeline.sh --image data/raw/photo.png --gpu 7 --max_steps 5000 --bg_method sam
```

参数：`--image` 输入图，`--bg_method` 去背景方案（`rembg` 快速 / `sam` 精细），`--max_steps` 训练步数，`--gpu` GPU 编号。

分步跑：

```bash
python remove_background.py --input data/raw/photo.png --output data/processed/photo_nobg.png --method rembg --crop
python train_image_to_3d.py --config configs/image-to-3d.yaml --image data/processed/photo_nobg.png --output_dir output --max_steps 5000 --gpu_id 7 --export_mesh
```

输出在 `third_party/threestudio/output/image-to-3d-zero123/<图片名>@<timestamp>/`，结构和物体 B 一样。导出 Mesh 方式也一样。

## 快速开始

```bash
conda activate 3d
export CUDA_HOME=$CONDA_PREFIX
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH
export HF_ENDPOINT=https://hf-mirror.com

cd object_a && bash run_pipeline.sh --video data/video.mp4 --gpu 7 --max_frames 100
cd object_b && bash run_pipeline.sh --prompt "a 3D model of a wooden chair" --max_steps 10000 --gpu 7
wget -O ~/.u2net/u2net.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx
cd object_c && bash run_pipeline.sh --image data/raw/photo.png --gpu 7 --max_steps 5000
```

## 常见问题

**COLMAP 跑不动？** 抽帧密一点（减小 `--interval`），拍的时候手稳一点，或者换 `--camera_model PINHOLE`。

**3DGS 爆显存？** 降分辨率 `--resolution 800`，或者少跑几轮 `--iterations 10000`。

**threestudio 生成质量差？** prompt 写详细点，多加几步（`--max_steps`），或者调配置文件里的 `guidance_scale`（50-150 试试）。

**rembg 下载 u2net.onnx 超时？** 手动下：`wget -O ~/.u2net/u2net.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx`

**Zero123 生成和原图差很多？** 用 `--bg_method sam` 做精细去背景，确保输入是正面照。

**输出不在 object_b/output/？** 脚本把工作目录切到了 threestudio 仓库，实际输出去 `third_party/threestudio/output/` 找。

## 性能参考

物体 A（3DGS）：~8-12 GB 显存，15-30 分钟 | 物体 B（SDS）：~10-15 GB，30-60 分钟 | 物体 C（Zero123）：~8-12 GB，20-40 分钟。都在 RTX 3090 上测的。

