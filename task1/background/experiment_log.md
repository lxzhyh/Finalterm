# Background Reconstruction Experiment Log

## 2026-06-20

- Goal: reconstruct the Mip-NeRF 360 `garden` scene as the task1 background using Graphdeco 3DGS.
- Data: `task1/background/data/garden`, 185 images with COLMAP `sparse/0`.
- GPU check: A800 80GB GPUs are visible from the interactive terminal environment. GPUs 0-3 have 0% utilization; GPU 3 had the most free memory at the time of checking.
- Repository setup: cloned `graphdeco-inria/gaussian-splatting` into `task1/third_party/gaussian-splatting`.
- Environment note: the interactive `conda activate hw3d_assets` shell resolves to `/usr/bin/python`, which has CUDA 12.8 PyTorch available. The absolute conda Python at `anaconda3/envs/hw3d_assets/bin/python` contains `torch 2.12.1+cu130` and cannot initialize CUDA on the current 12.8 driver.
- Dependency setup attempt 1:
  - Installed missing `plyfile` with user-site pip.
  - Built `simple_knn` and `fused_ssim` successfully.
  - `diff_gaussian_rasterization` failed to compile because `rasterizer_impl.h` missed `<cstdint>`.
- Fix:
  - Patched `submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h` to include `<cstdint>`.
  - Downgraded user-site NumPy to `1.26.4` and rebuilt `diff_gaussian_rasterization`.
  - Installed and verified `diff_gaussian_rasterization`, `simple_knn`, and `fused_ssim`.
- Debug training command:

```bash
cd task1/background
GS_REPO_PATH=/fudan_university_cfs/sjy/Finalterm/task1/third_party/gaussian-splatting \
python train_background_3dgs.py \
  --scene_path data/garden \
  --images images_4 \
  --output_path output/garden_debug \
  --iterations 7000 \
  --gpu_id 3
```

- Debug training log: `task1/background/logs/garden_debug_20260620.log`.
- Debug result:
  - Runtime: 6.7 min.
  - Initial COLMAP points: 138,766.
  - Iteration 7000 Gaussian count: 3,623,148.
  - Train metrics at 7000: L1 = 0.027991, PSNR = 27.5288 dB.
  - Output PLY: `task1/background/output/garden_debug/point_cloud/iteration_7000/point_cloud.ply` (857 MB).

- Full training command:

```bash
cd task1/background
GS_REPO_PATH=/fudan_university_cfs/sjy/Finalterm/task1/third_party/gaussian-splatting \
python train_background_3dgs.py \
  --scene_path data/garden \
  --images images_4 \
  --output_path output/garden \
  --iterations 30000 \
  --gpu_id 3 \
  --render
```

- Full training log: `task1/background/logs/garden_30000_20260620.log`.
- Full training result:
  - Runtime: 30.7 min for training; rendering ran after training.
  - Iteration 7000: L1 = 0.027976, PSNR = 27.5157 dB, Gaussians = 3,600,954, PLY size = 852 MB.
  - Iteration 15000: L1 = 0.021806, PSNR = 29.4717 dB, Gaussians = 4,184,254, PLY size = 990 MB.
  - Iteration 30000: L1 = 0.019991, PSNR = 30.2127 dB, Gaussians = 4,184,254, PLY size = 990 MB.
  - Final output PLY: `task1/background/output/garden/point_cloud/iteration_30000/point_cloud.ply`.
  - Render output: `task1/background/output/garden/train/ours_30000/renders`, 185 rendered training views.
  - Ground-truth render comparison directory: `task1/background/output/garden/train/ours_30000/gt`, 185 images.
  - Test render split is empty for this run (`test/ours_30000/renders` has 0 images), consistent with the current script/data split.

- Novel-view flythrough:
  - Added `task1/background/render_novel_view.py`.
  - Method: load the trained 3DGS checkpoint at iteration 30000, interpolate a closed camera path between the 185 COLMAP training camera poses, and render intermediate views with the Graphdeco CUDA rasterizer.
  - Command:

```bash
cd task1/background
GS_REPO_PATH=/fudan_university_cfs/sjy/Finalterm/task1/third_party/gaussian-splatting \
python render_novel_view.py \
  --model_path output/garden \
  --output_dir output/garden/novel_view/path_interp_30000 \
  --frames 240 \
  --width 1280 \
  --fps 30 \
  --gpu_id 3
```

  - Log: `task1/background/logs/garden_novel_view_20260620.log`.
  - Output frames: `task1/background/output/garden/novel_view/path_interp_30000/frames`, 240 PNG frames.
  - Output video: `task1/background/output/garden/novel_view/path_interp_30000/flythrough.mp4` (85 MB, 8 seconds at 30 fps).
  - Camera trajectory: `task1/background/output/garden/novel_view/path_interp_30000/trajectory.json`.
  - Smoke test output: `task1/background/output/garden/novel_view/smoke`.
