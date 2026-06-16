# 基于 3DGS 与 AIGC 的多源资产生成与真实场景融合 & 基于 LeRobot 的 ACT 策略跨环境泛化

<div align="center">

**小组成员**

| 姓名 | 学号 |
|---|---|
| 张天翼 | 25110980028 |
| {name_2} | {id_2} |
| {name_3} | {id_3} |

**GitHub 仓库**: https://github.com/lxzhyh/Finalterm  
**模型权重下载**:
1. 3D资产模型以及实验结果：https://drive.google.com/drive/folders/1JwwIzF0aPhRCudD8SY8ltvWElJ9PGZdQ?usp=sharing

---

# 题目一：基于 3DGS 与 AIGC 的多源资产生成与真实场景融合

## 1.1 实验概述

本题目旨在构建一条"全链路"3D 视觉流水线：从真实世界多视角重建与 AIGC 虚拟资产生成，到统一场景融合渲染。我们分别采用三条异构技术路线生成三个独立 3D 物体（物体 A / B / C），从开源数据集中选取并重建一个背景场景，最后将三者以合理比例和空间关系插入该背景，生成漫游渲染视频。

所有实验在一台搭载 NVIDIA GeForce RTX 3090 (24 GB) 的 Linux 服务器上完成，Python 3.10 + PyTorch 2.12 + CUDA 13.0。

---

## 1.2 3D 资产准备

以下三条技术路线分别代表了多视角几何重建、文本到 3D 跨模态生成、单图到 3D 推理三种核心范式。三者在输入模态、优化目标和 3D 表达上存在本质差异，对比具有方法论层面的价值。

| | 物体 A | 物体 B | 物体 C |
|---|---|---|---|
| **技术路线** | 多视角几何重建 | 文本到 3D 扩散生成 | 单图到 3D 扩散生成 |
| **核心框架** | COLMAP + 3DGS | threestudio + SDS | threestudio + Zero123 |
| **输入** | 环绕视频 (~30s) | 文本 Prompt | 单张 RGB 照片 |
| **先验来源** | 多视角几何一致性 | SD 2.1 扩散先验 | Zero123-XL 视角先验 |
| **3D 表达** | 显式 3D Gaussian 点云 | NeRF-like 隐式场 → Mesh | NeRF-like 隐式场 → Mesh |
| **训练时长** | ~30 min (3DGS 30k iters) | ~40 min (SDS 10k steps) | ~24 min (Zero123 5k steps) |
| **显存占用** | ~8 GB | ~14 GB | ~12 GB |

### 1.2.1 物体 A：多视角几何重建 (COLMAP + 3DGS)

#### 方法原理

物体 A 采用经典的"运动恢复结构 (SfM) + 可微渲染优化"路线。首先通过 COLMAP 从多视角图像中联合估计相机内/外参数与稀疏 3D 点云；随后以该稀疏点云为初始化，使用 3D Gaussian Splatting 对场景进行可微光栅化训练，优化每个高斯球的颜色、位置、协方差和不透明度参数。

3DGS 的核心思想是将场景表达为一组各向异性的 3D 高斯核 $\{(\mu_i, \Sigma_i, \alpha_i, c_i)\}$，通过基于 tile 的可微光栅化器将高斯核投影到图像平面，以逐像素 $L_1$ 和 SSIM 损失驱动优化：

$$
\mathcal{L} = (1 - \lambda) \mathcal{L}_1 + \lambda \mathcal{L}_{\mathrm{SSIM}}
$$

其显式表达使得训练收敛速度快于 NeRF，且天然支持实时渲染。

#### 数据采集与预处理

使用手机以 1080p 分辨率拍摄物体的环绕视频，时长约 30 秒，相机以物体为中心保持近似等距圆弧运动。

| 抽帧参数 | 取值 |
|---|---|
| 抽帧间隔 | 10 帧 |
| 最大图像边长 | 1280 px |
| 实际抽取帧数 | 100（从 ~1560 帧的 4K@60fps 视频中抽取） |

#### COLMAP SfM 流程

1. **特征提取**：SIFT (CPU 模式避免无头服务器 OpenGL 问题)，每帧最多 8192 个特征点。
2. **特征匹配**：sequential matcher，邻帧 20 张，开启交叉验证和几何验证。
3. **稀疏重建**：OPENCV 相机模型（支持径向+切向畸变），所有帧共享同一相机内参。
4. **图像去畸变**：输出至 `dense/` 目录供 3DGS 使用。

#### 3DGS 训练超参数

| 超参数 | 取值 |
|---|---|
| 迭代次数 | 30,000 |
| SH 阶数 | 3 |
| 位置学习率 (init/final) | $1.6\times10^{-4}$ / $1.6\times10^{-6}$ |
| 不透明度学习率 | 0.05 |
| 缩放学习率 | 0.005 |
| 旋转学习率 | 0.001 |
| 特征学习率 | 0.0025 |
| 稠密化间隔 | 100 steps |
| 稠密化范围 | 500–15,000 steps |
| 稠密化梯度阈值 | $2\times10^{-4}$ |
| SSIM 权重 $\lambda$ | 0.2 |

#### 重建效果

3DGS 在 30,000 次迭代后收敛良好，生成的 3D Gaussian 点云在训练视角下渲染结果与原始照片高度一致。测试视角下细节保持良好，无明显孔洞或漂浮噪声。完整的训练/测试渲染图、checkpoint（7,000 / 15,000 / 30,000 iterations）和 input.ply 保存在 `object_a/output/` 下。

![物体 A 渲染结果 1](assets/a_1.png)

![物体 A 渲染结果 2](assets/a_2.png)

---

### 1.2.2 物体 B：文本到 3D 扩散生成 (threestudio + SDS)

#### 方法原理

物体 B 基于 **Score Distillation Sampling (SDS)** 范式。给定文本描述 $y$，SDS 利用预训练 2D 文本到图像扩散模型 $\epsilon_\phi$ 作为可微"渲染质量评判器"，将 3D 表达的可微渲染结果 $g(\theta)$ 注入扩散模型，通过 Score 蒸馏梯度驱动 3D 参数 $\theta$ 更新：

$$
\nabla_\theta \mathcal{L}_{\mathrm{SDS}} = \mathbb{E}_{t, \epsilon} \left[ w(t) (\epsilon_\phi(x_t; y, t) - \epsilon) \frac{\partial g}{\partial \theta} \right]
$$

其中 $x_t = \alpha_t g(\theta) + \sigma_t \epsilon$。选用 Stable Diffusion 2.1-base 作为 2D 扩散先验，3D 表达使用 threestudio 中的 Mip-NeRF 360 隐式辐射场，并引入法向光滑正则项抑制浮空伪影。

#### 超参数

| 超参数 | 取值 |
|---|---|
| 最大训练步数 | 10,000 |
| 扩散先验 | Stable Diffusion 2.1-base |
| 3D 表达 | Mip-NeRF 360 隐式场 |
| 渲染分辨率 | 64×64（训练）/ 512×512（测试） |
| 优化器 | AdamW |
| 学习率 | $1\times10^{-2}$ (geometry & appearance) |
| 混合精度 | fp16 |
| 视角采样 | 随机方位角 + 固定仰角范围 |

#### 输入 Prompt

| Prompt | 目标描述 |
|---|---|
| "a high quality 3D render of a cute baby dragon, white background" | 一只可爱的卡通小龙 |
| "a 3D model of a wooden chair" | 一把木质椅子 |

#### 生成效果

椅子（"a 3D model of a wooden chair"）成功完成约 9,800 步训练（接近 max_steps=10,000），生成结果具有可辨识的椅子结构，四条腿和座面清晰可辨。

![物体 B 生成结果](assets/b.png)

---

### 1.2.3 物体 C：单图到 3D 扩散生成 (Zero123 + threestudio)

#### 方法原理

物体 C 从单张 2D 照片推理完整 3D 几何，分两步：

**第一步：背景去除。** 使用 SAM (Segment Anything Model, ViT-H) 或 rembg (U²-Net) 提取纯净前景物体，可按 alpha 边界框裁剪。

**第二步：Zero123 引导的 3D 优化。** Zero123 是 Stable Diffusion 在新视角合成任务上的微调变体。在 SDS 框架下，条件从纯文本变为输入图像 $I_{\mathrm{ref}}$ + 相对位姿 $\Delta R$：

$$
\nabla_\theta \mathcal{L}_{\mathrm{SDS-Zero123}} = \mathbb{E}_{t, \epsilon, \Delta R} \left[ w(t) (\epsilon_\phi(x_t; I_{\mathrm{ref}}, \Delta R, t) - \epsilon) \frac{\partial g}{\partial \theta} \right]
$$

每次迭代随机采样一个相机视角，Zero123 推断该视角下的"应有外观"，梯度反向传播更新 3D 表达。同时引入深度平滑正则和法向一致性约束抑制 Janus 多面体问题。

#### 超参数

| 超参数 | 取值 |
|---|---|
| 最大训练步数 | 5,000 |
| 扩散先验 | Zero123-XL |
| 输入图像分辨率 | 256×256 |
| 渲染分辨率 | 64×64（训练）/ 512×512（测试） |
| 背景去除方案 | SAM |
| 优化器 | AdamW |
| 混合精度 | fp16 |
| 相机采样 | 仰角 0°–30°，方位角 −180°–180° |
| 正则项 | 深度平滑 + 法向一致性 |

#### 生成效果

输入为一张真实拍摄的日常物品照片（经 SAM 去背景处理为 photo_nobg.png），训练在 5,000 步收敛，输出 Mesh 可从 it5000-export 导出。从验证视频来看，物体在大部分方位角上保持了与输入图像一致的外观，背面和侧面存在一定程度的模糊和几何退化，符合单图到 3D 方法的典型表现。完整输出（含各步 checkpoint 和验证渲染）保存在 `output/image-to-3d-zero123/photo_nobg.png@20260616-201341/` 下。

![物体 C 正面视角](assets/c_front.png)

![物体 C 左侧视角](assets/c_left.png)

---

### 1.2.4 三种资产准备方法对比

| | 物体 A (多视角重建) | 物体 B (文本到 3D) | 物体 C (单图到 3D) |
|---|---|---|---|
| 几何准确度 | 高——多视角几何约束保证精确重建 | 中——SDS 生成存在浮空伪影与结构偏差 | 中——单图歧义导致背面几何不可靠 |
| 纹理细节 | 高——真实照片纹理，细节丰富 | 中——扩散先验产生合理但模糊的纹理 | 中——Zero123 推断纹理在输入视角附近较好，背面退化 |
| 数据准备耗时 | ~15 min（拍摄+COLMAP SfM） | 0（仅需文本） | ~5 min（拍照+rembg 去背景） |
| 训练耗时 | ~30 min | ~40 min | ~24 min |
| Mesh 导出耗时 | —（显式表达，无需导出） | ~5 min（Marching Cubes 提取 Mesh） | ~3 min（Marching Cubes 提取 Mesh） |
| **总耗时** | ~45 min | ~45 min | ~32 min |
| 适用场景 | 高保真实物复刻 | 概念/创意生成 | 便捷 3D 建模 |

---

## 1.3 背景场景重建

<!-- TODO: 选择 Mip-NeRF 360 场景，COLMAP+3DGS 重建，填入效果与指标 -->
（待补充）

---

## 1.4 场景融合与渲染

<!-- TODO: 异构表达统一、物体摆放、漫游相机路径、融合渲染结果 -->
（待补充）

---

## 1.5 题目一质量评估与技术报告

<!-- TODO: 三种方法对比分析、训练过程可视化（WandB/SwanLab）、技术讨论 -->
（待补充）

---

<br>
<br>

# 题目二：基于 LeRobot 的 ACT 策略跨环境泛化挑战

<!-- TODO: CALVIN 数据集、ACT 方法、实验一（仅A训练）、实验二（A+B+C联合训练）、实验三（环境D Zero-shot泛化）、结果分析与总结 -->
（待补充）

---

<br>
<br>
