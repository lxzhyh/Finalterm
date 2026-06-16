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

### 1. 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate hw3d_assets
```

或手动安装：

```bash
conda create -n hw3d_assets python=3.10
conda activate hw3d_assets
pip install -r requirements.txt
```

### 2. 安装 COLMAP

**Ubuntu/Debian:**
```bash
sudo apt-get install colmap
```

**从源码编译:**
```bash
git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build && cd build
cmake .. -DCUDA_ENABLED=ON
make -j$(nproc)
sudo make install
```
conda update -n base -c defaults conda
conda install -c nvidia cuda-toolkit=13.0
export CUDA_HOME=$CONDA_PREFIX

conda install colmap
conda install faiss
conda install openimageio

# CUDA toolkit（如需要）
conda install -c nvidia cuda-toolkit=13.0
export CUDA_HOME=$CONDA_PREFIX
```

### 3. 安装 3D Gaussian Splatting

```bash
cd ../third_party  # 或你选择的目录
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting

# 安装子模块
sed -i '1i #include <cstdint>' submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h
pip install --no-build-isolation submodules/diff-gaussian-rasterization
pip install --no-build-isolation submodules/simple-knn

# 安装主包
pip install plyfile tqdm
```

设置环境变量（可选）：
```bash
export GS_REPO_PATH=/path/to/gaussian-splatting
```

### 4. 安装 threestudio

```bash
# 克隆仓库
cd ~/homework/Finalterm/task1/third_party
git clone https://github.com/threestudio-project/threestudio.git
cd threestudio

# 安装 nvdiffrast
git clone https://github.com/NVlabs/nvdiffrast.git
pip install --no-build-isolation -e nvdiffrast

# 安装依赖
pip install --no-build-isolation -r requirements.txt
pip install pybind11
pip install --no-build-isolation pysdf
CXXFLAGS="-std=c++14" pip install --no-build-isolation nerfacc==0.5.2

# 安装 threestudio
pip install --no-build-isolation -e .
```

设置环境变量（可选）：
```bash
export THREESTUDIO_REPO_PATH=/path/to/threestudio
```
git clone --recursive https://github.com/NVlabs/tiny-cuda-nn.git



### 5. 环境变量（运行时必须设置）

```bash
# CUDA 13 头文件路径修复（threestudio 的 nerfacc JIT 编译必需）
export CUDA_HOME=$CONDA_PREFIX
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH

# HuggingFace 镜像（国内网络必需）
export HF_ENDPOINT=https://hf-mirror.com
```

以上变量已写入各 `run_pipeline.sh` 中，直接运行脚本即可。

### 6. 下载预训练模型

**Stable Diffusion (用于物体 B):**
```bash
# 模型会在首次训练时自动下载（HF_ENDPOINT 已设为 hf-mirror）
hf download Manojb/stable-diffusion-2-1-base --local-dir ./models/sd-2-1-base
```

**Zero123 (用于物体 C):**
```bash
# 模型会在首次训练时自动下载
hf download bennyguo/zero123-xl-diffusers --local-dir ./models/zero123-xl
```

**rembg u2net 模型（用于物体 C 背景去除）：**
```bash
# 首次运行 rembg 时需要从 GitHub 下载，可能超时
# 建议提前手动下载到 ~/.u2net/u2net.onnx：
wget -O ~/.u2net/u2net.onnx \
  https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx
```

**SAM (可选，用于更精确的背景分割):**
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P ~/models/
```

---

## 物体 A: 真实多视角重建

使用手机拍摄真实物体的环绕视频，通过 COLMAP 提取相机位姿，再用 3DGS 进行高斯溅射重建。

### 数据准备

1. 使用手机拍摄物体的环绕视频（建议 30-60 秒，保持物体静止，相机环绕）
2. 将视频文件放置在任意位置，例如 `object_a/data/video.mp4`

### 运行完整流水线

```bash
cd object_a
bash run_pipeline.sh --video data/video.mp4 --gpu 7 --max_frames 100
```

**参数说明:**
- `--video <path>`: 输入视频路径（必填）
- `--interval <int>`: 抽帧间隔，默认 2（每隔 2 帧取 1 帧）
- `--max_frames <int>`: 最大帧数，默认 300
- `--iterations <int>`: 3DGS 训练迭代次数，默认 30000
- `--resolution <int>`: 图像分辨率，默认 -1（使用原始分辨率）
- `--gpu <int>`: GPU 设备 ID，默认 0

**示例:**
```bash
bash run_pipeline.sh --video data/my_object.mp4 --interval 3 --max_frames 200 --iterations 20000
```

### 分步执行

如果需要更细粒度的控制，可以分步执行：

```bash
# Step 1: 从视频抽帧
python extract_frames.py --video data/video.mp4 --output_dir data/images --interval 2

# Step 2: COLMAP SfM
python run_colmap.py --image_dir data/images --workspace data --gpu_id 0

# Step 3: 3DGS 训练
python train_3dgs.py --source_path data/dense --output_path output --iterations 30000 --gpu_id 7 --render
```

### 输出结果

- `data/images/`: 抽取的帧图像
- `data/sparse/`: COLMAP 稀疏重建结果
- `data/dense/`: 去畸变后的图像和相机参数
- `output/`: 3DGS 模型（包含 point_cloud.ply）

---

## 物体 B: 文本到 3D 生成

使用 threestudio 框架，基于 Stable Diffusion 和 SDS Loss，从文本描述生成 3D 物体。

### 运行生成

```bash
cd object_b
bash run_pipeline.sh --prompt "a 3D model of a wooden chair" --max_steps 10000 --gpu 7
```

**参数说明:**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `configs/text-to-3d.yaml` | 配置文件路径 |
| `--prompt` | （配置文件中定义） | 文本描述 |
| `--output_dir` | `output` | 输出目录（实际输出在 threestudio 仓库内） |
| `--max_steps` | `10000` | 最大训练步数 |
| `--gpu` | `0` | GPU 设备 ID |

### 输出结果

训练输出实际保存在 **threestudio 仓库**的 `output/` 目录下（非 `object_b/output/`）：

```
third_party/threestudio/output/text-to-3d-sds/<prompt>@<timestamp>/
├── ckpts/       ← 模型 checkpoint（last.ckpt）
├── save/        ← 测试结果 & 训练过程渲染图
├── configs/     ← 完整配置
├── tb_logs/     ← TensorBoard 日志
└── csv_logs/    ← 训练日志
```

### 导出 Mesh

训练完成后，通过 threestudio `--export` 导出带纹理的 OBJ 网格：

```bash
cd third_party/threestudio
python launch.py \
  --config "output/text-to-3d-sds/<你的实验目录>/configs/parsed.yaml" \
  --export \
  --gpu 7 \
  "resume=output/text-to-3d-sds/<你的实验目录>/ckpts/last.ckpt"
```

导出结果在 `<实验目录>/save/it<步数>-export/` 下：`model.obj` + `model.mtl` + `texture_kd.jpg`。

---

## 物体 C: 单图到 3D 生成

使用 Zero123 从单张图像生成 3D 模型。首先去除图像背景，再通过 threestudio + Zero123 扩散模型生成。

### 数据准备

1. 拍摄一张物体的清晰照片（建议正面视角，背景简洁）
2. 将图像放置在 `object_c/data/raw/` 目录下（已有一张示例 `photo.png`）

### 运行完整流水线

```bash
pip install 'mediapipe'
cd object_c
bash run_pipeline.sh --image data/raw/photo.png --gpu 0 --max_steps 5000 --bg_method sam
```

**参数说明:**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image` | （必填） | 输入图像路径 |
| `--config` | `configs/image-to-3d.yaml` | 配置文件路径 |
| `--output_dir` | `output` | 输出目录 |
| `--max_steps` | `5000` | 最大训练步数 |
| `--gpu` | `0` | GPU 设备 ID |
| `--bg_method` | `rembg` | 背景去除方法：`rembg` 或 `sam` |

### 分步执行

```bash
# Step 1: 去除背景
python remove_background.py \
    --input data/raw/photo.png \
    --output data/processed/photo_nobg.png \
    --method rembg \
    --crop

# Step 2: Zero123 生成
python train_image_to_3d.py \
    --config configs/image-to-3d.yaml \
    --image data/processed/photo_nobg.png \
    --output_dir output \
    --max_steps 5000 \
    --gpu_id 7 \
    --export_mesh
```

### 输出结果

训练输出实际保存在 **threestudio 仓库**的 `output/` 目录下：

```
third_party/threestudio/output/image-to-3d-zero123/<图片名>@<timestamp>/
├── ckpts/       ← 模型 checkpoint
├── save/        ← 测试结果
├── configs/     ← 完整配置
└── csv_logs/    ← 训练日志
```

### 导出 Mesh

训练完成后，用 threestudio `--export` 导出 OBJ 网格（同 Object B 导出方式）。

---

## 快速开始（复制即用）

```bash
# 激活环境 + 设置变量
conda activate 3d
export CUDA_HOME=$CONDA_PREFIX
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH
export HF_ENDPOINT=https://hf-mirror.com

# Object A: 真实多视角重建
cd object_a && bash run_pipeline.sh --video data/video.mp4 --gpu 7 --max_frames 100

# Object B: 文本到 3D 生成
cd object_b && bash run_pipeline.sh --prompt "a 3D model of a wooden chair" --max_steps 10000 --gpu 7

# Object C: 单图到 3D 生成（先下载 rembg 模型）
wget -O ~/.u2net/u2net.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx
cd object_c && bash run_pipeline.sh --image data/raw/photo.png --gpu 7 --max_steps 5000
```

## 常见问题

### Q1: COLMAP 重建失败

**可能原因:**
- 图像质量差、模糊
- 视角变化过大或过小
- 特征点不足

**解决方案:**
- 增加抽帧密度（减小 `--interval`）
- 确保拍摄时物体静止，相机平稳移动
- 尝试不同的相机模型（`--camera_model PINHOLE`）

### Q2: 3DGS 训练显存不足

**解决方案:**
- 降低图像分辨率：`--resolution 800`
- 减少迭代次数：`--iterations 10000`
- 使用更小的 batch size（修改 `train_3dgs.py`）

### Q3: threestudio 生成质量不佳

**可能原因:**
- Prompt 描述不清晰
- 训练步数不足
- Guidance scale 设置不当

**解决方案:**
- 优化 Prompt，添加更多细节描述
- 增加 `--max_steps`
- 调整配置文件中的 `guidance_scale`（建议 50-150）

### Q4: rembg 下载 u2net.onnx 超时

**原因:** GitHub 连接超时，无法自动下载。

**解决方案:**
```bash
wget -O ~/.u2net/u2net.onnx \
  https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx
```

### Q5: Zero123 生成结果与输入图像差异大

**可能原因:**
- 背景去除不干净
- 输入图像视角不佳
- 训练步数不足

**解决方案:**
- 使用 SAM 进行更精确的背景分割：`--bg_method sam`
- 确保输入图像是正面视角
- 增加 `--max_steps`

### Q6: threestudio 训练输出不在 object_b/output/

**原因:** `train_text_to_3d.py`/`train_image_to_3d.py` 将工作目录切换到 threestudio 仓库，`exp_root_dir=output` 相对于仓库目录。

**实际路径:** `third_party/threestudio/output/` 下找对应实验目录。

---

## 性能参考

| 方法 | 显存需求 | 训练时间 (RTX 3090) | 输出格式 |
|------|---------|-------------------|---------|
| 物体 A (3DGS) | 8-12 GB | 15-30 分钟 | 3D Gaussians (.ply) |
| 物体 B (SDS) | 10-15 GB | 30-60 分钟 | NeRF / Mesh |
| 物体 C (Zero123) | 8-12 GB | 20-40 分钟 | NeRF / Mesh |

---

## 下一步

完成三个物体的生成后，需要进行：

1. **背景场景重建**: 使用 3DGS 重建 Mip-NeRF 360 数据集中的场景
2. **场景融合**: 将三个物体插入到背景场景中
3. **渲染视频**: 生成多视角漫游渲染视频

这些步骤将在后续任务中实现。

---

## 参考资料

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [COLMAP](https://colmap.github.io/)
- [threestudio](https://github.com/threestudio-project/threestudio)
- [Zero123](https://zero123.cs.columbia.edu/)
- [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
- [rembg](https://github.com/danielgatis/rembg)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
