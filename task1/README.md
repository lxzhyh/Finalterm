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
├── background/            # Mip-NeRF 360 garden 背景 3DGS 重建
│   ├── train_background_3dgs.py
│   ├── render_novel_view.py
│   ├── run_pipeline.sh
│   ├── data/
│   ├── output/
│   └── logs/
├── fusion/                # 背景与 A/B/C 的 Gaussian 融合
│   ├── configs/
│   ├── gaussian_utils.py
│   ├── mesh_to_gaussians.py
│   ├── merge_gaussians.py
│   ├── render_flythrough.py
│   ├── output/
│   └── logs/
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
cd YOUR_PROJCT_ROOT/task1/third_party
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

训练中间文件由 threestudio 生成；最终用于提交与融合的导出 Mesh 统一整理到 `task1/object_b/output/` 下。里面应包含导出的 `model.obj`、`model.mtl`、`texture_kd.jpg`，以及后续转换得到的 Gaussian PLY。

导出 Mesh：

```bash
cd third_party/threestudio
python launch.py \
  --config "output/text-to-3d-sds/<你的实验目录>/configs/parsed.yaml" \
  --export --gpu 7 \
  "resume=output/text-to-3d-sds/<你的实验目录>/ckpts/last.ckpt"
```

导出产物包括 `model.obj` + `model.mtl` + `texture_kd.jpg`。完成导出后，将这些文件复制到 `task1/object_b/output/` 供融合阶段使用。

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

训练中间文件由 threestudio 生成；最终导出的 Mesh 统一整理到 `task1/object_c/output/` 下，结构和物体 B 类似。导出 Mesh 方式也一样。

## 背景场景重建

背景使用 Mip-NeRF 360 的 `garden` 场景，数据放在：

```bash
task1/background/data/garden/
```

该数据已经包含 COLMAP 格式的 `sparse/0`，可以直接用 Graphdeco 3DGS 训练。建议使用 `hw3d_assets` 环境，并显式指定 3DGS 仓库路径：

```bash
cd task1/background
GS_REPO_PATH=YOUR_PROJCT_ROOT/task1/third_party/gaussian-splatting \
python train_background_3dgs.py \
  --scene_path data/garden \
  --images images_4 \
  --output_path output/garden \
  --iterations 30000 \
  --gpu_id 3 \
  --render
```

快速调试可以先跑 7000 次迭代：

```bash
cd task1/background
GS_REPO_PATH=YOUR_PROJCT_ROOT/task1/third_party/gaussian-splatting \
python train_background_3dgs.py \
  --scene_path data/garden \
  --images images_4 \
  --output_path output/garden_debug \
  --iterations 7000 \
  --gpu_id 3
```

训练完成后可生成背景自由视角视频：

```bash
cd task1/background
GS_REPO_PATH=YOUR_PROJCT_ROOT/task1/third_party/gaussian-splatting \
python render_novel_view.py \
  --model_path output/garden \
  --output_dir output/garden/novel_view/path_interp_30000 \
  --frames 240 \
  --width 1280 \
  --fps 30 \
  --gpu_id 3
```

本次实验记录：

- 30,000 次迭代训练耗时约 30.7 min。
- 最终训练视角指标：L1 = 0.019991，PSNR = 30.2127 dB。
- 最终背景模型包含 4,184,254 个 Gaussians。

## 场景融合与渲染

融合模块在 `task1/fusion/`。统一表达路线如下：

- 背景：Graphdeco 3DGS Gaussian PLY
- 物体 A：Graphdeco 3DGS Gaussian PLY
- 物体 B/C：threestudio 导出的 Mesh，先采样为 Gaussian PLY
- 最终合并为一个 Graphdeco 字段兼容的 `point_cloud.ply`

### 1. 准备已有 A/B/C 资产

若使用已经打包好的 `object_abc.zip`，先解压到临时目录，再将 A/B/C 资产整理到各自的 `object_*/output` 下：

```bash
cd YOUR_PROJCT_ROOT
mkdir -p /tmp/object_abc task1/object_a/output task1/object_b/output task1/object_c/output
unzip -o /path/to/object_abc.zip -d /tmp/object_abc

cp -r /tmp/object_abc/object_a/output/* task1/object_a/output/
cp -r /tmp/object_abc/object_b/output/* task1/object_b/output/
cp -r /tmp/object_abc/object_c/output/* task1/object_c/output/
```

其中 A 已经是 3DGS PLY，B/C 是 OBJ + texture。

### 2. 将 B/C Mesh 转成 Gaussian PLY

```bash
cd task1/fusion
python mesh_to_gaussians.py \
  --mesh ../object_b/output/it10000-export/model.obj \
  --output ../object_b/output/object_b_mesh_gaussians.ply \
  --samples 100000 \
  --gaussian_scale 0.01 \
  --opacity 0.85 \
  --normalize \
  --seed 20

python mesh_to_gaussians.py \
  --mesh ../object_c/output/mesh/model.obj \
  --output ../object_c/output/object_c_mesh_gaussians.ply \
  --samples 100000 \
  --gaussian_scale 0.01 \
  --opacity 0.85 \
  --normalize \
  --seed 30
```

### 3. 调整摆放配置

最终融合使用：

```bash
task1/fusion/configs/scene_layout_object_abc.yaml
```

主要调每个物体的：

- `scale`
- `rotation_deg`
- `translation`

### 4. 合并场景

```bash
cd task1/fusion
python merge_gaussians.py \
  --config configs/scene_layout_object_abc.yaml \
  --output_model output/fused_scene_object_abc \
  --iteration 30000
```

最终融合场景包含：

| 组成部分 | Gaussian 数量 |
|---|---:|
| background garden | 4,184,254 |
| object A | 93,225 |
| object B | 100,000 |
| object C | 100,000 |
| fused scene | 4,477,479 |

### 5. 高质量 3DGS 渲染

渲染原始训练相机视角：

```bash
cd task1/third_party/gaussian-splatting
CUDA_VISIBLE_DEVICES=3 python render.py \
  -m YOUR_PROJCT_ROOT/task1/fusion/output/fused_scene_object_abc \
  --iteration 30000
```

渲染自由视角视频：

```bash
cd task1/background
GS_REPO_PATH=YOUR_PROJCT_ROOT/task1/third_party/gaussian-splatting \
python render_novel_view.py \
  --model_path ../fusion/output/fused_scene_object_abc \
  --output_dir ../fusion/output/fused_scene_object_abc/novel_view/path_interp_30000 \
  --frames 240 \
  --width 1280 \
  --fps 30 \
  --gpu_id 3
```

本次实验输出了 185 张原始相机视角渲染图和 240 帧自由视角视频，自由视角渲染速度约 3.0 frames/s。

### 6. 快速点云预览

`render_flythrough.py` 是轻量级点云 splat 预览渲染器，用来快速检查物体尺度、位置和相机轨迹。最终高质量结果仍以 Graphdeco renderer 为准。

```bash
cd task1/fusion
bash run_pipeline.sh --config configs/scene_layout_object_abc.yaml
```

## 快速开始

```bash
conda activate hw3d_assets
export CUDA_HOME=$CONDA_PREFIX
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH
export HF_ENDPOINT=https://hf-mirror.com

cd task1/background
GS_REPO_PATH=YOUR_PROJCT_ROOT/task1/third_party/gaussian-splatting \
python train_background_3dgs.py --scene_path data/garden --images images_4 --output_path output/garden --iterations 30000 --gpu_id 3 --render

cd ../fusion
python merge_gaussians.py --config configs/scene_layout_object_abc.yaml --output_model output/fused_scene_object_abc --iteration 30000
```

## 常见问题

**COLMAP 跑不动？** 抽帧密一点（减小 `--interval`），拍的时候手稳一点，或者换 `--camera_model PINHOLE`。

**3DGS 爆显存？** 降分辨率 `--resolution 800`，或者少跑几轮 `--iterations 10000`。

**threestudio 生成质量差？** prompt 写详细点，多加几步（`--max_steps`），或者调配置文件里的 `guidance_scale`（50-150 试试）。

**rembg 下载 u2net.onnx 超时？** 手动下：`wget -O ~/.u2net/u2net.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx`

**Zero123 生成和原图差很多？** 用 `--bg_method sam` 做精细去背景，确保输入是正面照。

**threestudio 中间文件在哪里？** 训练中间文件通常在 threestudio 仓库的 `output/` 下；最终提交与融合使用的资产需要整理到 `task1/object_b/output/` 或 `task1/object_c/output/`。

**B/C mesh 转 Gaussian 报 `trimesh` 缺失？** 在当前环境中安装：`pip install trimesh`。

**CUDA 在非交互 shell 中不可用？** 可用 `bash -i -c 'conda activate hw3d_assets && python -c "import torch; print(torch.cuda.is_available())"'` 检查环境。

## 性能参考

物体 A（3DGS）：~8-12 GB 显存，15-30 分钟 | 物体 B（SDS）：~10-15 GB，30-60 分钟 | 物体 C（Zero123）：~8-12 GB，20-40 分钟。都在 RTX 3090 上测的。

本次 garden 背景 3DGS 训练 30,000 次迭代约 30.7 分钟；A/B/C 与背景融合约 15 秒；融合场景自由视角 240 帧渲染约 79 秒。
