# NeRF: Neural Radiance Fields

A from-scratch PyTorch reproduction of [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) (Mildenhall et al., ECCV 2020).

Supports both **Blender Synthetic** (360° object-centric) and **LLFF** (forward-facing real-world) datasets, including NDC space warping and spiral render path generation.

## Results

### Blender Synthetic (Lego) & LLFF (Fern)

<table>
<tr>
<td align="center"><b>Lego</b> (400×400, factor=2, 200k iters, ~30 dB)</td>
<td align="center"><b>Fern</b> (756×1008, factor=4, 200k iters, ~26 dB)</td>
</tr>
<tr>
<td align="center">

<!-- TODO: Replace with actual video URL from GitHub Issue -->
https://github.com/YOUR_USER/YOUR_REPO/assets/VIDEO_LEGO_F2

</td>
<td align="center">

<!-- TODO: Replace with actual video URL from GitHub Issue -->
https://github.com/YOUR_USER/YOUR_REPO/assets/VIDEO_FERN_F4

</td>
</tr>
</table>

### Full Resolution (Lego 800×800)

<!-- TODO: Replace with actual video URL from GitHub Issue -->
https://github.com/YOUR_USER/YOUR_REPO/assets/VIDEO_LEGO_F1

> Lego scene at full 800×800 resolution, 200k iterations, ~33 dB PSNR.

### Training Summary

| Scene | Dataset | Resolution | Factor | Iterations | PSNR | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Lego | Blender | 400×400 | 2 | 200,000 | ~30 dB | ~8h |
| Lego | Blender | 800×800 | 1 | 200,000 | ~33 dB | ~17h |
| Fern | LLFF | 756×1008 | 4 | 200,000 | ~26 dB | ~15h |

## Lessons Learned

### Cross Product Order Matters

When constructing orthonormal camera frames from averaged axes, the cross product order must follow the **xyz positive cycle**: `y × z = x`, `z × x = y`, `x × y = z`. Reversing the operand order (e.g., `cross(z, y)` instead of `cross(y, z)`) produces a **reflected** coordinate frame (`det(R) = -1`), resulting in a mirrored/flipped scene:

<table>
<tr>
<td align="center"><b>Correct</b></td>
<td align="center"><b>Wrong cross product order</b></td>
</tr>
<tr>
<td align="center">

<!-- TODO: Replace with actual video URL from GitHub Issue -->
https://github.com/YOUR_USER/YOUR_REPO/assets/VIDEO_FERN_CORRECT

</td>
<td align="center">

<!-- TODO: Replace with actual video URL from GitHub Issue -->
https://github.com/YOUR_USER/YOUR_REPO/assets/VIDEO_FERN_WRONG

</td>
</tr>
</table>

### Spiral Render Path: Direction & Target

For forward-facing scenes, the spiral camera path must:
1. **Place the look-at target in front of the camera** (along `-z`), not behind it. The c2w matrix column 2 stores the backwards direction (`+z`), so the target should be `position - backwards * depth`, not `+`.
2. **Transform offsets through the average pose's rotation**, not add them directly in world coordinates. Otherwise the spiral plane won't align with the camera's viewing plane.

Getting either of these wrong produces distorted or flipped renderings.

### Dataset Integrity

If training shows no convergence at all (e.g., PSNR stuck at ~5 dB for tens of thousands of steps while the same code works on other scenes), consider **re-downloading the dataset**. A corrupted `poses_bounds.npy` or missing/mismatched images can cause complete training failure with no obvious error message.

## Project Structure

```
NeRF_Mildenhall2020_ECCV/
├── configs/
│   ├── lego.yaml             # Blender synthetic scene config
│   └── fern.yaml             # LLFF real-world scene config
├── data/
│   ├── load_blender.py       # NeRF Synthetic dataset loader
│   └── load_llff.py          # LLFF dataset loader (with NDC support)
├── models/
│   ├── embedder.py           # Positional encoding (Fourier features)
│   └── nerf.py               # NeRF MLP architecture
├── rendering/
│   ├── ray_marching.py       # Ray generation, NDC, sampling, volume rendering
│   └── render_video.py       # Video generation from trained model
├── training/
│   └── trainer.py            # Training loop orchestration
├── utils/
│   └── metrics.py            # PSNR metric
├── main.py                   # Entry point (training & rendering)
└── requirements.txt
```

## Quick Start

### 1. Environment Setup

```bash
conda create -n nerf python=3.10
conda activate nerf
pip install -r requirements.txt
```

### 2. Prepare Dataset

**Blender Synthetic** (e.g., Lego): Download from [NeRF Synthetic dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).

**LLFF** (e.g., Fern): Download from [NeRF LLFF dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).

Update the `datadir` field in the corresponding config file:

```yaml
data:
  datadir: "/path/to/your/dataset"
```

### 3. Train

```bash
# Blender synthetic scene
python main.py --config configs/lego.yaml

# LLFF real-world scene
python main.py --config configs/fern.yaml
```

### 4. Render Video

```bash
# Render from checkpoint at step 200000
python main.py --config configs/lego.yaml --render_only --ckpt_step 200000
python main.py --config configs/fern.yaml --render_only --ckpt_step 200000
```

## Key Implementation Details

- **Coarse-to-Fine Sampling**: 64 stratified coarse samples + 128 importance-weighted fine samples per ray.
- **Positional Encoding**: 10 frequency bands for position (xyz), 4 for viewing direction.
- **MLP Architecture**: 8-layer, 256-wide density branch with skip connection at layer 4; 128-wide color branch conditioned on view direction.
- **NDC Space Warping**: For forward-facing (LLFF) scenes, rays are warped into Normalized Device Coordinates to handle unbounded depth.
- **Training**: Adam optimizer, initial LR 5e-4 with exponential decay.

## Reference

```bibtex
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
  booktitle={ECCV},
  year={2020}
}
```
