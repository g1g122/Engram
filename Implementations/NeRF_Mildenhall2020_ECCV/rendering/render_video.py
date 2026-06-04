"""
Video rendering module for trained NeRF models.
Generates a 360-degree or spiral rendering video using test/render poses.
Supports both Blender (synthetic) and LLFF (real-world) datasets.
"""

import os
import time

import imageio
import numpy as np
import torch
import yaml

from data.load_blender import load_blender_data
from data.load_llff import load_llff_data
from rendering.ray_marching import get_rays, ndc_rays, render_rays
from training.trainer import Trainer


def generate_360_video(cfg_path, step, out_name="video_360.mp4"):
    """Loads a trained model and renders a 360-degree video.
    
    Args:
        cfg_path: Path to the yaml config file.
        step: the checkpoint iteration step to load (e.g., 200000).
        out_name: Output filename for the mp4 video.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Preparing Video Rendering on {device} ---")

    # 1. Load Config
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Load data to extract render_poses
    dataset_type = cfg["data"].get("dataset_type", "blender")
    print(f"Loading {dataset_type} data to extract render_poses...")

    if dataset_type == "llff":
        _, _, render_poses, hwf, _ = load_llff_data(
            basedir=cfg["data"]["datadir"],
            factor=cfg["data"].get("factor", 8),
        )
    else:
        _, _, render_poses, hwf, _ = load_blender_data(
            basedir=cfg["data"]["datadir"],
            factor=cfg["data"].get("factor", 1),
            testskip=cfg["data"].get("testskip", 1)
        )

    H, W, focal = int(hwf[0]), int(hwf[1]), hwf[2]
    N_poses = render_poses.shape[0]
    print(f"Successfully loaded {N_poses} render poses.")

    # 3. Initialize Trainer architecture (to host our loaded weights)
    trainer = Trainer(cfg, device=device)
    
    # 4. Load heavily-trained Brain (Checkpoint)
    save_dir = os.path.join(cfg["experiment"]["base_dir"], cfg["experiment"]["name"])
    ckpt_path = os.path.join(save_dir, f"ckpt_{step:06d}.pt")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Did you finish training?")
        
    trainer.load_checkpoint(ckpt_path)
    
    # Switch models to eval mode (good practice, though our MLP doesn't use dropout/batchnorm)
    trainer.coarse_model.eval()
    trainer.fine_model.eval()

    # 5. Start Chunked Rendering
    # Creating output directory
    out_dir = os.path.join(save_dir, "renders")
    os.makedirs(out_dir, exist_ok=True)
    
    # Extract rendering hyperparameters
    chunk = cfg["rendering"]["chunk"]
    near, far = cfg["rendering"]["near"], cfg["rendering"]["far"]
    N_samples = cfg["rendering"]["N_samples"]
    N_importance = cfg["rendering"]["N_importance"]
    white_bkgd = cfg["data"].get("white_bkgd", False)
    use_ndc = cfg["rendering"].get("use_ndc", False)

    # For NDC scenes, override bounds to [0, 1]
    if use_ndc:
        near, far = 0.0, 1.0
    
    frames = []
    
    print(f"Starting rendering... (Total frames: {N_poses})")
    start_t = time.time()
    
    with torch.no_grad():
        for i, c2w in enumerate(render_poses):
            c2w = torch.tensor(c2w, dtype=torch.float32)
            rays_o, rays_d = get_rays(H, W, focal, c2w)

            # Apply NDC warping for forward-facing scenes
            if use_ndc:
                rays_o, rays_d = ndc_rays(H, W, focal, 1.0, rays_o, rays_d)
            
            # Flatten rays to feed into chunked pipeline
            rays_o_flat = rays_o.reshape(-1, 3).to(device)
            rays_d_flat = rays_d.reshape(-1, 3).to(device)
            
            rgb_map_flat = []
            
            # Process in chunks to prevent OOM
            for j in range(0, rays_o_flat.shape[0], chunk):
                ro = rays_o_flat[j : j + chunk]
                rd = rays_d_flat[j : j + chunk]
                
                # CRITICAL: perturb=False to prevent temporal flickering/noise
                res = render_rays(
                    ro, rd,
                    near=near, far=far,
                    N_samples=N_samples, N_importance=N_importance,
                    coarse_model=trainer.coarse_model,
                    fine_model=trainer.fine_model,
                    embed_pos_fn=trainer.embed_pos,
                    embed_dir_fn=trainer.embed_dir,
                    perturb=False,  
                    white_bkgd=white_bkgd,
                    chunk=chunk
                )
                rgb_map_flat.append(res["rgb_map"].cpu())
                
            # Reconstruct the full image
            rgb_map_flat = torch.cat(rgb_map_flat, dim=0)
            rgb_img = rgb_map_flat.reshape(H, W, 3).numpy()
            
            # Convert float [0,1] to uint8 [0,255] for standard video formats
            img8 = (np.clip(rgb_img, 0, 1) * 255.0).astype(np.uint8)
            frames.append(img8)
            
            print(f"Rendered frame [{i+1}/{N_poses}]")

    print(f"All frames rendered in {(time.time() - start_t)/60:.2f} minutes.")
    
    # 6. Export to MP4
    out_path = os.path.join(out_dir, out_name)
    print(f"Compressing into video: {out_path}")
    imageio.mimwrite(out_path, frames, fps=30, quality=8)
    print("Done!")
