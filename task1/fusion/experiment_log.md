# Task1 Scene Fusion Experiment Log

Date: 2026-06-20

## Goal

Complete the scene fusion and rendering stage after the Garden background 3DGS reconstruction. The target is to insert object A/B/C into the reconstructed background using a unified Gaussian representation, then render multi-view and free-viewpoint results.

## Inputs

- Background 3DGS model:
  - `../background/output/garden/point_cloud/iteration_30000/point_cloud.ply`
  - Background training metrics at iteration 30000: L1 `0.019991`, PSNR `30.2127`, Gaussian count `4,184,254`
- Existing object preview assets reused for this run:
  - A: `../../assets/a_1.png`
  - B: `../../assets/b.png`
  - C: `../../assets/c_front.png`

Note: the repository currently does not contain final A/B/C trained mesh or Gaussian PLY outputs. To keep the experiment runnable end-to-end, this run converts the existing object preview images into lightweight Gaussian billboard assets. These files are placeholders for the final A/B/C assets and can be replaced directly in `configs/scene_layout.yaml`.

## Object Gaussian Assets

Generated with `image_to_gaussians.py`:

```bash
bash -i -c 'conda activate hw3d_assets && cd /fudan_university_cfs/sjy/Finalterm/task1/fusion && python image_to_gaussians.py --image ../../assets/a_1.png --output output/object_a_gaussians.ply --samples 45000 --width_world 0.75 --depth 0.05 --gaussian_scale 0.012 --opacity 0.85 --bg_tolerance 28 --seed 1'
bash -i -c 'conda activate hw3d_assets && cd /fudan_university_cfs/sjy/Finalterm/task1/fusion && python image_to_gaussians.py --image ../../assets/b.png --output output/object_b_gaussians.ply --samples 35000 --width_world 0.55 --depth 0.04 --gaussian_scale 0.011 --opacity 0.85 --bg_tolerance 26 --seed 2'
bash -i -c 'conda activate hw3d_assets && cd /fudan_university_cfs/sjy/Finalterm/task1/fusion && python image_to_gaussians.py --image ../../assets/c_front.png --output output/object_c_gaussians.ply --samples 35000 --width_world 0.55 --depth 0.04 --gaussian_scale 0.011 --opacity 0.85 --bg_tolerance 24 --seed 3'
```

Outputs:

- `output/object_a_gaussians.ply`: 45,000 Gaussians
- `output/object_b_gaussians.ply`: 35,000 Gaussians
- `output/object_c_gaussians.ply`: 35,000 Gaussians

## Scene Layout

Layout file: `configs/scene_layout.yaml`

- A: translation `[-0.85, 1.95, 0.15]`, scale `1.0`
- B: translation `[0.0, 1.95, 0.15]`, scale `1.0`
- C: translation `[0.85, 1.95, 0.15]`, scale `1.0`

These positions place the three inserted objects around the garden table region.

## Fusion

Command:

```bash
bash -i -c 'conda activate hw3d_assets && cd /fudan_university_cfs/sjy/Finalterm/task1/fusion && mkdir -p logs && python merge_gaussians.py --config configs/scene_layout.yaml --output_model output/fused_scene --iteration 30000 2>&1 | tee -a logs/fusion_20260620.log'
```

Output:

- Fused model: `output/fused_scene/point_cloud/iteration_30000/point_cloud.ply`
- Total Gaussian count: `4,299,254`
- Model size: about `1017M`

## Rendering On Original Cameras

Command:

```bash
bash -i -c 'conda activate hw3d_assets && cd /fudan_university_cfs/sjy/Finalterm/task1/third_party/gaussian-splatting && CUDA_VISIBLE_DEVICES=3 python render.py -m /fudan_university_cfs/sjy/Finalterm/task1/fusion/output/fused_scene --iteration 30000 2>&1 | tee -a /fudan_university_cfs/sjy/Finalterm/task1/fusion/logs/fused_render_train_20260620.log'
```

Output:

- Rendered train-camera frames: `output/fused_scene/train/ours_30000/renders`
- Frame count: `185`
- Representative checked frames:
  - `00000.png`: A/B/C visible around the front of the table
  - `00059.png`: side view shows billboard-thin behavior for current placeholder object assets

## Free-Viewpoint Flythrough

Command:

```bash
bash -i -c 'conda activate hw3d_assets && cd /fudan_university_cfs/sjy/Finalterm/task1/background && GS_REPO_PATH=/fudan_university_cfs/sjy/Finalterm/task1/third_party/gaussian-splatting python render_novel_view.py --model_path ../fusion/output/fused_scene --output_dir ../fusion/output/fused_scene/novel_view/path_interp_30000 --frames 240 --width 1280 --fps 30 --gpu_id 3 2>&1 | tee -a ../fusion/logs/fused_novel_view_20260620.log'
```

Output:

- Frames: `output/fused_scene/novel_view/path_interp_30000/frames`
- Frame count: `240`
- Video: `output/fused_scene/novel_view/path_interp_30000/flythrough.mp4`, about `84M`
- Camera trajectory: `output/fused_scene/novel_view/path_interp_30000/trajectory.json`
- Render speed: about `2.9` frames/s for 1280px-wide output on GPU 3

## Current Limitations

- A/B/C are generated from existing preview images, not final task-specific 3D reconstructions/generations.
- Because the current object assets are thin Gaussian billboards, they look acceptable from front-facing views but become visibly flat from oblique views.
- The fusion/rendering code path is complete. For the final report-grade run, replace:
  - `output/object_a_gaussians.ply`
  - `output/object_b_gaussians.ply`
  - `output/object_c_gaussians.ply`
  with the final object A/B/C Gaussian or mesh-to-Gaussian outputs, then rerun `merge_gaussians.py` and the render commands above.

## Rerun With Provided Object A/B/C Assets

Date: 2026-06-20

The provided asset archive `/fudan_university_cfs/sjy/object_abc.zip` was extracted to:

- `assets/object_abc/object_a/output/point_cloud/iteration_30000/point_cloud.ply`
- `assets/object_abc/object_b/output/it10000-export/model.obj`
- `assets/object_abc/object_b/output/it10000-export/texture_kd.jpg`
- `assets/object_abc/object_c/output/mesh/model.obj`
- `assets/object_abc/object_c/output/mesh/texture_kd.jpg`

Object A already contains a 3DGS point cloud and was used directly. Object B/C are textured OBJ meshes, so they were converted to Gaussian PLY files with the existing mesh conversion script:

```bash
bash -i -c 'conda activate hw3d_assets && cd /fudan_university_cfs/sjy/Finalterm/task1/fusion && python mesh_to_gaussians.py --mesh assets/object_abc/object_b/output/it10000-export/model.obj --output output/object_b_mesh_gaussians.ply --samples 100000 --gaussian_scale 0.01 --opacity 0.85 --normalize --seed 20'
bash -i -c 'conda activate hw3d_assets && cd /fudan_university_cfs/sjy/Finalterm/task1/fusion && python mesh_to_gaussians.py --mesh assets/object_abc/object_c/output/mesh/model.obj --output output/object_c_mesh_gaussians.ply --samples 100000 --gaussian_scale 0.01 --opacity 0.85 --normalize --seed 30'
```

Object statistics:

- A: `93,225` Gaussians from the provided 3DGS PLY
- B: `100,000` mesh-sampled Gaussians
- C: `100,000` mesh-sampled Gaussians

Layout file:

- `configs/scene_layout_object_abc.yaml`

Fusion command:

```bash
bash -i -c 'conda activate hw3d_assets && cd /fudan_university_cfs/sjy/Finalterm/task1/fusion && mkdir -p logs && python merge_gaussians.py --config configs/scene_layout_object_abc.yaml --output_model output/fused_scene_object_abc --iteration 30000 2>&1 | tee -a logs/fusion_object_abc_20260620.log'
```

Fusion output:

- Fused model: `output/fused_scene_object_abc/point_cloud/iteration_30000/point_cloud.ply`
- Total Gaussian count: `4,477,479`
- Fused output directory size: about `2.4G`

Original-camera rendering command:

```bash
bash -i -c 'conda activate hw3d_assets && cd /fudan_university_cfs/sjy/Finalterm/task1/third_party/gaussian-splatting && CUDA_VISIBLE_DEVICES=3 python render.py -m /fudan_university_cfs/sjy/Finalterm/task1/fusion/output/fused_scene_object_abc --iteration 30000 2>&1 | tee -a /fudan_university_cfs/sjy/Finalterm/task1/fusion/logs/fused_object_abc_render_train_20260620.log'
```

Original-camera rendering output:

- Frames: `output/fused_scene_object_abc/train/ours_30000/renders`
- Frame count: `185`
- Checked frames: `00000.png`, `00059.png`
- Observation: B/C show clear volumetric mesh-derived shapes; A is visible but sparse/noisy because its provided 3DGS point cloud has limited object density and transparent regions.

Free-viewpoint rendering command:

```bash
bash -i -c 'conda activate hw3d_assets && cd /fudan_university_cfs/sjy/Finalterm/task1/background && GS_REPO_PATH=/fudan_university_cfs/sjy/Finalterm/task1/third_party/gaussian-splatting python render_novel_view.py --model_path ../fusion/output/fused_scene_object_abc --output_dir ../fusion/output/fused_scene_object_abc/novel_view/path_interp_30000 --frames 240 --width 1280 --fps 30 --gpu_id 3 2>&1 | tee -a ../fusion/logs/fused_object_abc_novel_view_20260620.log'
```

Free-viewpoint rendering output:

- Frames: `output/fused_scene_object_abc/novel_view/path_interp_30000/frames`
- Frame count: `240`
- Video: `output/fused_scene_object_abc/novel_view/path_interp_30000/flythrough.mp4`, about `85M`
- Camera trajectory: `output/fused_scene_object_abc/novel_view/path_interp_30000/trajectory.json`
- Render speed: about `3.0` frames/s for 1280px-wide output on GPU 3
