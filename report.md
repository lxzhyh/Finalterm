# 基于 3DGS 与 AIGC 的多源资产生成与真实场景融合 & 基于 LeRobot 的 ACT 策略跨环境泛化

<div align="center">

**小组成员**

| 姓名 | 学号 |
|---|---|
| 张天翼 | 25110980028 |
| {name_2} | {id_2} |
| {name_3} | {id_3} |

**GitHub 仓库**: https://github.com/lxzhyh/Finalterm  
**模型权重下载**: {model_weights_url}（提取码: {pwd}）

</div>

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
| **训练时长** | {A_time} | {B_time} | {C_time} |
| **显存占用** | {A_vram} | {B_vram} | {C_vram} |

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
| 抽帧间隔 | {A_frame_interval} |
| 最大图像边长 | 1280 px |
| 实际抽取帧数 | {A_num_frames} |

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

<!-- TODO: 填入效果图、PSNR/SSIM 指标等 -->
{placeholder_A_results}

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
| {B_prompt_1} | {B_desc_1} |
| {B_prompt_2} | {B_desc_2} |

#### 生成效果

<!-- TODO: 填入效果图、生成结果分析 -->
{placeholder_B_results}

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
| 背景去除方案 | {C_bg_method} |
| 优化器 | AdamW |
| 混合精度 | fp16 |
| 相机采样 | 仰角 0°–30°，方位角 −180°–180° |
| 正则项 | 深度平滑 + 法向一致性 |

#### 生成效果

<!-- TODO: 填入效果图、生成结果分析 -->
{placeholder_C_results}

---

### 1.2.4 三种资产准备方法对比

| | 物体 A (多视角重建) | 物体 B (文本到 3D) | 物体 C (单图到 3D) |
|---|---|---|---|
| 几何准确度 | {A_geo} | {B_geo} | {C_geo} |
| 纹理细节 | {A_tex} | {B_tex} | {C_tex} |
| 数据准备耗时 | {A_prep_time} | 0（仅需文本） | {C_prep_time} |
| 训练耗时 | {A_train_time} | {B_train_time} | {C_train_time} |
| Mesh 导出耗时 | —（显式表达） | {B_export_time} | {C_export_time} |
| **总耗时** | {A_total} | {B_total} | {C_total} |
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
