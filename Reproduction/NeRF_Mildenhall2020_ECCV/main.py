"""
NeRF: Neural Radiance Fields — Main Entry Point.

Usage:
    # Training (Blender synthetic)
    python main.py --config configs/lego.yaml

    # Training (LLFF real-world)
    python main.py --config configs/fern.yaml

    # Render final video from checkpoint
    python main.py --config configs/lego.yaml --render_only --ckpt_step 200000

Reference: Mildenhall et al. 2020, ECCV.
"""

import argparse

import yaml
import torch

from data.load_blender import load_blender_data
from data.load_llff import load_llff_data
from training.trainer import Trainer
from rendering.render_video import generate_360_video


def main():
    parser = argparse.ArgumentParser(description="Train or render a NeRF model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file."
    )
    parser.add_argument(
        "--render_only", action="store_true", help="If set, skip training and render a 360 video from checkpoint."
    )
    parser.add_argument(
        "--ckpt_step", type=int, default=200000, help="The checkpoint step to load when rendering."
    )
    args = parser.parse_args()

    if args.render_only:
        # Jump directly to rendering video
        generate_360_video(args.config, step=args.ckpt_step)
        return

    # --- Training Mode ---
    # Load configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset based on type
    dataset_type = cfg["data"].get("dataset_type", "blender")
    print(f"Loading {dataset_type} data from: {cfg['data']['datadir']}")

    if dataset_type == "llff":
        images, poses, render_poses, hwf, i_split = load_llff_data(
            basedir=cfg["data"]["datadir"],
            factor=cfg["data"].get("factor", 8),
        )
    else:
        images, poses, render_poses, hwf, i_split = load_blender_data(
            basedir=cfg["data"]["datadir"],
            factor=cfg["data"].get("factor", 1),
            testskip=cfg["data"].get("testskip", 1),
        )
    print(f"Images: {images.shape}, Poses: {poses.shape}")
    print(f"H={int(hwf[0])}, W={int(hwf[1])}, focal={hwf[2]:.2f}")
    print(f"Train: {len(i_split[0])}, Val: {len(i_split[1])}, Test: {len(i_split[2])}")

    # Initialize trainer and start training
    trainer = Trainer(cfg, device=device)
    trainer.train(images, poses, hwf, i_split)


if __name__ == "__main__":
    main()
