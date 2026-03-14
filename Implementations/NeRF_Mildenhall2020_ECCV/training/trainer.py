"""
Training orchestration for NeRF.

Manages the complete training lifecycle: model instantiation, data loading,
ray batch sampling, loss computation, optimization, checkpoint saving,
and periodic validation logging.

Reference: Mildenhall et al. 2020, Section 5.3.
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.embedder import get_embedder
from models.nerf import NeRF
from rendering.ray_marching import get_rays, ndc_rays, render_rays
from utils.metrics import mse2psnr


class Trainer:
    """NeRF training loop manager.

    Args:
        cfg: Configuration dictionary loaded from YAML.
        device: Torch device ('cuda' or 'cpu').
    """

    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device

        self._build_models()
        self._build_optimizer()

    def _build_models(self):
        """Instantiate embedders and coarse/fine NeRF networks."""
        cfg_model = self.cfg["model"]

        self.embed_pos, pos_dim = get_embedder(cfg_model["pos_L"], in_dims=3)
        self.embed_dir, dir_dim = get_embedder(cfg_model["dir_L"], in_dims=3)
        self.embed_pos.to(self.device)
        self.embed_dir.to(self.device)

        model_kwargs = dict(
            pos_dim=pos_dim,
            dir_dim=dir_dim,
            net_width=cfg_model["net_width"],
            color_width=cfg_model["color_width"],
            net_depth=cfg_model["net_depth"],
            skip_layer=cfg_model["skip_layer"],
        )
        self.coarse_model = NeRF(**model_kwargs).to(self.device)
        self.fine_model = NeRF(**model_kwargs).to(self.device)

    def _build_optimizer(self):
        """Set up Adam optimizer with all trainable parameters."""
        cfg_train = self.cfg["training"]
        params = (
            list(self.coarse_model.parameters())
            + list(self.fine_model.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=cfg_train["lrate"])

    def _update_learning_rate(self, step):
        """Exponential learning rate decay."""
        cfg_train = self.cfg["training"]
        decay_rate = cfg_train["lrate_decay_rate"]
        decay_steps = cfg_train["lrate_decay_steps"]
        new_lr = cfg_train["lrate"] * (decay_rate ** (step / decay_steps))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr

    def train(self, images, poses, hwf, i_split):
        """Run the full training loop.

        Args:
            images: All images, shape [N, H, W, 4], float32 RGBA in [0, 1].
            poses: Camera-to-world matrices, shape [N, 4, 4].
            hwf: [H, W, focal].
            i_split: [i_train, i_val, i_test] index arrays.
        """
        cfg_render = self.cfg["rendering"]
        cfg_train = self.cfg["training"]
        cfg_log = self.cfg["logging"]

        H, W, focal = int(hwf[0]), int(hwf[1]), hwf[2]
        i_train, i_val, i_test = i_split
        near, far = cfg_render["near"], cfg_render["far"]
        white_bkgd = self.cfg["data"].get("white_bkgd", False)
        use_ndc = cfg_render.get("use_ndc", False)

        # For NDC scenes, sampling bounds are always [0, 1]
        if use_ndc:
            near, far = 0.0, 1.0

        # Precompute all training rays for efficient random sampling
        all_rays_o, all_rays_d, all_target_rgb = self._precompute_rays(
            images, poses, H, W, focal, i_train, white_bkgd, use_ndc
        )

        N_total_rays = all_rays_o.shape[0]
        print(f"Precomputed {N_total_rays:,} training rays.")

        # Training loop
        start_time = time.time()
        for step in range(1, cfg_train["N_iters"] + 1):
            # --- Sample a random batch of rays ---
            batch_idx = np.random.randint(0, N_total_rays, cfg_train["batch_size"])
            batch_rays_o = all_rays_o[batch_idx].to(self.device)
            batch_rays_d = all_rays_d[batch_idx].to(self.device)
            batch_target = all_target_rgb[batch_idx].to(self.device)

            # --- Forward: render rays through coarse-to-fine pipeline ---
            result = render_rays(
                batch_rays_o,
                batch_rays_d,
                near=near,
                far=far,
                N_samples=cfg_render["N_samples"],
                N_importance=cfg_render["N_importance"],
                coarse_model=self.coarse_model,
                fine_model=self.fine_model,
                embed_pos_fn=self.embed_pos,
                embed_dir_fn=self.embed_dir,
                perturb=True,
                white_bkgd=white_bkgd,
                chunk=cfg_render["chunk"],
            )

            # --- Compute loss: MSE on both coarse and fine outputs ---
            loss_fine = F.mse_loss(result["rgb_map"], batch_target)
            loss_coarse = F.mse_loss(result["rgb_map_coarse"], batch_target)
            loss = loss_fine + loss_coarse

            # --- Backward + optimize ---
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # --- Learning rate decay ---
            new_lr = self._update_learning_rate(step)

            # --- Logging ---
            if step % cfg_log["print_every"] == 0:
                psnr = mse2psnr(loss_fine.item())
                elapsed = time.time() - start_time
                print(
                    f"Step {step:>6d}/{cfg_train['N_iters']} | "
                    f"Loss: {loss.item():.5f} | "
                    f"PSNR: {psnr:.2f} dB | "
                    f"LR: {new_lr:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )

            # --- Save checkpoint ---
            if step % cfg_log["save_every"] == 0:
                self._save_checkpoint(step)

        print("Training complete.")

    def _precompute_rays(self, images, poses, H, W, focal, i_train, white_bkgd, use_ndc=False):
        """Precompute all training rays and target colors.

        Flattens all training images into a single pool of rays for
        efficient random batch sampling during training.

        Args:
            use_ndc: If True, warp rays into NDC space for forward-facing scenes.

        Returns:
            rays_o: [N_total, 3]
            rays_d: [N_total, 3]
            target_rgb: [N_total, 3]
        """
        all_rays_o = []
        all_rays_d = []
        all_target = []

        for i in i_train:
            c2w = torch.tensor(poses[i], dtype=torch.float32)
            rays_o, rays_d = get_rays(H, W, focal, c2w)

            # Warp rays into NDC space for forward-facing scenes
            if use_ndc:
                rays_o, rays_d = ndc_rays(H, W, focal, 1.0, rays_o, rays_d)

            # Extract RGB target (apply alpha compositing if white background)
            img = torch.tensor(images[i], dtype=torch.float32)
            if white_bkgd and img.shape[-1] == 4:
                rgb = img[..., :3] * img[..., 3:] + (1.0 - img[..., 3:])
            else:
                rgb = img[..., :3]

            all_rays_o.append(rays_o.reshape(-1, 3))
            all_rays_d.append(rays_d.reshape(-1, 3))
            all_target.append(rgb.reshape(-1, 3))

        return (
            torch.cat(all_rays_o, dim=0),
            torch.cat(all_rays_d, dim=0),
            torch.cat(all_target, dim=0),
        )

    def _save_checkpoint(self, step):
        """Save model weights and optimizer state."""
        save_dir = os.path.join(
            self.cfg["experiment"]["base_dir"],
            self.cfg["experiment"]["name"],
        )
        os.makedirs(save_dir, exist_ok=True)

        ckpt_path = os.path.join(save_dir, f"ckpt_{step:06d}.pt")
        torch.save(
            {
                "step": step,
                "coarse_model": self.coarse_model.state_dict(),
                "fine_model": self.fine_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            ckpt_path,
        )
        print(f"Checkpoint saved: {ckpt_path}")

    def load_checkpoint(self, ckpt_path):
        """Restore model weights and optimizer state from a checkpoint."""
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.coarse_model.load_state_dict(ckpt["coarse_model"])
        self.fine_model.load_state_dict(ckpt["fine_model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Loaded checkpoint from step {ckpt['step']}: {ckpt_path}")
        return ckpt["step"]
