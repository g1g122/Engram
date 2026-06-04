"""
Ray Marching and Volume Rendering for NeRF.

Implements the complete rendering pipeline:
  1. Ray generation from camera parameters (get_rays).
  2. NDC space warping for forward-facing scenes (ndc_rays).
  3. Stratified sampling along rays (sample_stratified).
  4. Volume rendering integration (raw2outputs).
  5. Hierarchical importance sampling (sample_pdf).
  6. Full render pipeline tying everything together (render_rays).

Coordinate Convention:
    Blender / OpenGL: +X Right, +Y Up, -Z Forward (camera looks along -Z).

Reference: Mildenhall et al. 2020, Sections 4 & 5.
"""

import torch
import torch.nn.functional as F


def get_rays(H, W, focal, c2w):
    """Generate rays for every pixel in an image.

    Each ray is defined by an origin (camera center in world coordinates)
    and a direction (pixel direction transformed to world coordinates).

    Args:
        H: Image height in pixels.
        W: Image width in pixels.
        focal: Focal length in pixels.
        c2w: Camera-to-world matrix, shape [4, 4] or [3, 4].

    Returns:
        rays_o: Ray origins in world space, shape [H, W, 3].
        rays_d: Ray directions in world space (not normalized), shape [H, W, 3].
    """
    # Build a grid of pixel coordinates
    # i spans columns (x), j spans rows (y)
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=c2w.device),
        torch.arange(H, dtype=torch.float32, device=c2w.device),
        indexing="xy",
    )

    # Convert pixel coordinates to camera-space ray directions.
    # Pinhole model: direction = ((x - cx)/f, -(y - cy)/f, -1)
    # The negations follow OpenGL convention where Y is up and camera looks along -Z.
    dirs_cam = torch.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)],
        dim=-1,
    )  # [H, W, 3]

    # Rotate camera-space directions into world space: dir_world = R @ dir_cam
    # c2w[:3, :3] is the rotation matrix whose columns are the camera axes in world space.
    rays_d = torch.sum(dirs_cam[..., None, :] * c2w[:3, :3], dim=-1)  # [H, W, 3]

    # All rays originate from the camera center, which is the translation column of c2w.
    rays_o = c2w[:3, -1].expand_as(rays_d)  # [H, W, 3]

    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Warp rays from world space into Normalized Device Coordinate (NDC) space.

    For forward-facing scenes (e.g., LLFF data), the physical depth extends
    to infinity. NDC maps the frustum [near, ∞) onto the unit cube [0, 1]
    using a perspective projection, so that uniform sampling in NDC
    automatically concentrates more samples near the camera.

    After this transform, set near=0.0 and far=1.0 for stratified sampling.

    Math (from NeRF supplementary, Eq. 18-25):
        o'_x = -(2f / W) * (o_x / o_z)
        o'_y = -(2f / H) * (o_y / o_z)
        o'_z = 1 + 2*near / o_z
        (analogous formulas for d')

    Args:
        H: Image height in pixels.
        W: Image width in pixels.
        focal: Focal length in pixels.
        near: Near clipping plane distance (typically 1.0 after scene scaling).
        rays_o: Ray origins in world space, shape [..., 3].
        rays_d: Ray directions in world space, shape [..., 3].

    Returns:
        rays_o_ndc: Ray origins in NDC space, shape [..., 3].
        rays_d_ndc: Ray directions in NDC space, shape [..., 3].
    """
    # Shift ray origins to the near plane: t_near = -(near + o_z) / d_z
    t_near = -(near + rays_o[..., 2:3]) / rays_d[..., 2:3]
    rays_o = rays_o + t_near * rays_d

    # Project origin into NDC
    o0 = -2.0 * focal / W * (rays_o[..., 0:1] / rays_o[..., 2:3])
    o1 = -2.0 * focal / H * (rays_o[..., 1:2] / rays_o[..., 2:3])
    o2 = 1.0 + 2.0 * near / rays_o[..., 2:3]

    # Project direction into NDC
    d0 = -2.0 * focal / W * (rays_d[..., 0:1] / rays_d[..., 2:3] - rays_o[..., 0:1] / rays_o[..., 2:3])
    d1 = -2.0 * focal / H * (rays_d[..., 1:2] / rays_d[..., 2:3] - rays_o[..., 1:2] / rays_o[..., 2:3])
    d2 = -2.0 * near / rays_o[..., 2:3]

    rays_o_ndc = torch.cat([o0, o1, o2], dim=-1)
    rays_d_ndc = torch.cat([d0, d1, d2], dim=-1)

    return rays_o_ndc, rays_d_ndc


def sample_stratified(rays_o, rays_d, near, far, N_samples, perturb=True):
    """Sample points along rays using stratified (jittered) sampling.

    Divides each ray into N_samples uniform bins and draws one random sample
    per bin, producing an unbiased estimate of the continuous volume integral.

    Args:
        rays_o: Ray origins, shape [N_rays, 3].
        rays_d: Ray directions, shape [N_rays, 3].
        near: Near clipping distance (scalar).
        far: Far clipping distance (scalar).
        N_samples: Number of samples per ray.
        perturb: If True, add uniform noise within each bin (training).
                 If False, sample at bin centers (evaluation).

    Returns:
        pts: Sampled 3D points, shape [N_rays, N_samples, 3].
        z_vals: Depth values along each ray, shape [N_rays, N_samples].
    """
    N_rays = rays_o.shape[0]

    # Uniform partition of [near, far] into N_samples bins
    t_vals = torch.linspace(0.0, 1.0, N_samples, device=rays_o.device)
    z_vals = near * (1.0 - t_vals) + far * t_vals  # [N_samples]
    z_vals = z_vals.expand(N_rays, N_samples).clone()  # [N_rays, N_samples]

    if perturb:
        # Width of each bin
        mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mid, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mid], dim=-1)
        # Uniform random offset within each bin
        z_vals = lower + (upper - lower) * torch.rand_like(z_vals)

    # Compute 3D positions: r(t) = o + t * d
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    return pts, z_vals


def raw2outputs(raw, z_vals, rays_d, white_bkgd=False):
    """Convert raw MLP outputs to rendered RGB color and auxiliary maps.

    Implements the discrete volume rendering equation:
        C = Σ T_i · α_i · c_i
    where α_i = 1 - exp(-σ_i · δ_i) and T_i = Π_{j<i} (1 - α_j).

    Args:
        raw: MLP output, shape [N_rays, N_samples, 4] — (R, G, B, σ).
        z_vals: Depth values, shape [N_rays, N_samples].
        rays_d: Ray directions, shape [N_rays, 3]. Used to scale δ by ray length.
        white_bkgd: If True, composite onto a white background.

    Returns:
        Dictionary containing:
            rgb_map: Rendered color, shape [N_rays, 3].
            depth_map: Expected depth, shape [N_rays].
            acc_map: Accumulated opacity, shape [N_rays].
            weights: Per-sample contribution weights, shape [N_rays, N_samples].
    """
    # MLP already applies sigmoid (RGB) and relu (sigma) in its forward pass.
    rgb = raw[..., :3]    # [N_rays, N_samples, 3]
    sigma = raw[..., 3]   # [N_rays, N_samples]

    # Compute distances between adjacent samples (δ_i = t_{i+1} - t_i)
    dists = z_vals[..., 1:] - z_vals[..., :-1]  # [N_rays, N_samples-1]
    # Pad the last distance with a large value (infinity sentinel)
    dists = torch.cat(
        [dists, torch.full_like(dists[..., :1], 1e10)], dim=-1
    )  # [N_rays, N_samples]

    # Scale distances by ray direction magnitude (accounts for non-unit directions)
    dists = dists * rays_d.norm(dim=-1, keepdim=True)  # [N_rays, N_samples]

    # Opacity: α_i = 1 - exp(-σ_i · δ_i)
    alpha = 1.0 - torch.exp(-sigma * dists)  # [N_rays, N_samples]

    # Transmittance: T_i = Π_{j<i} (1 - α_j) = exp(-Σ_{j<i} σ_j · δ_j)
    # Implemented via exclusive cumulative product of (1 - α)
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[..., :-1]  # [N_rays, N_samples]

    # Per-sample contribution weight: w_i = T_i · α_i
    weights = transmittance * alpha  # [N_rays, N_samples]

    # Rendered color: C = Σ w_i · c_i
    rgb_map = (weights[..., None] * rgb).sum(dim=-2)  # [N_rays, 3]

    # Expected depth: D = Σ w_i · t_i
    depth_map = (weights * z_vals).sum(dim=-1)  # [N_rays]

    # Accumulated opacity (for background compositing)
    acc_map = weights.sum(dim=-1)  # [N_rays]

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return {
        "rgb_map": rgb_map,
        "depth_map": depth_map,
        "acc_map": acc_map,
        "weights": weights,
    }


def sample_pdf(bins, weights, N_importance, perturb=True):
    """Importance-sample new depth values from a piecewise-constant PDF.

    Used in hierarchical sampling: the coarse network's weights define a PDF
    over the ray, and this function draws additional samples concentrated
    in high-weight (high-density) regions.

    Args:
        bins: Bin edges from coarse sampling, shape [N_rays, N_bins+1].
        weights: Per-bin weights from coarse rendering, shape [N_rays, N_bins].
        N_importance: Number of new samples to draw.
        perturb: If True, apply stratified sampling on the CDF.

    Returns:
        samples: New depth values, shape [N_rays, N_importance].
    """
    # Prevent division by zero
    weights = weights + 1e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)  # Normalize to PDF
    cdf = torch.cumsum(pdf, dim=-1)                     # [N_rays, N_bins]
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # Prepend 0

    # Draw uniform samples on [0, 1]
    if perturb:
        u = torch.rand(
            list(cdf.shape[:-1]) + [N_importance], device=cdf.device
        )
    else:
        u = torch.linspace(0.0, 1.0, N_importance, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])

    # Invert the CDF via binary search (searchsorted)
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)

    # Gather CDF values at the bounding indices
    cdf_below = torch.gather(cdf, -1, below)
    cdf_above = torch.gather(cdf, -1, above)
    bins_below = torch.gather(bins, -1, below)
    bins_above = torch.gather(bins, -1, above)

    # Linear interpolation within each bin
    denom = cdf_above - cdf_below
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_below) / denom
    samples = bins_below + t * (bins_above - bins_below)

    return samples


def render_rays(
    rays_o,
    rays_d,
    near,
    far,
    N_samples,
    N_importance,
    coarse_model,
    fine_model,
    embed_pos_fn,
    embed_dir_fn,
    perturb=True,
    white_bkgd=False,
    chunk=1024 * 32,
):
    """Render a batch of rays through the coarse-to-fine NeRF pipeline.

    Args:
        rays_o: Ray origins, shape [N_rays, 3].
        rays_d: Ray directions, shape [N_rays, 3].
        near: Near bound (scalar).
        far: Far bound (scalar).
        N_samples: Number of coarse samples per ray.
        N_importance: Number of additional fine samples per ray.
        coarse_model: Coarse NeRF MLP.
        fine_model: Fine NeRF MLP.
        embed_pos_fn: Positional encoding function for coordinates.
        embed_dir_fn: Positional encoding function for directions.
        perturb: Enable stochastic sampling (training) or deterministic (eval).
        white_bkgd: Composite onto white background.
        chunk: Maximum number of points to query MLP at once (memory control).

    Returns:
        Dictionary with 'rgb_map', 'depth_map', 'acc_map' for the fine network,
        plus 'rgb_map_coarse' for the coarse network.
    """
    # --- Stage 1: Coarse sampling & rendering ---
    pts_coarse, z_vals_coarse = sample_stratified(
        rays_o, rays_d, near, far, N_samples, perturb
    )

    raw_coarse = _run_network(
        pts_coarse, rays_d, coarse_model, embed_pos_fn, embed_dir_fn, chunk
    )

    coarse_outputs = raw2outputs(raw_coarse, z_vals_coarse, rays_d, white_bkgd)

    # --- Stage 2: Hierarchical importance sampling ---
    # Build bin edges from coarse z_vals for PDF sampling
    z_vals_mid = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
    z_samples = sample_pdf(
        z_vals_mid, coarse_outputs["weights"][..., 1:-1], N_importance, perturb
    )
    z_samples = z_samples.detach()

    # Merge coarse and fine samples, then sort by depth
    z_vals_fine, _ = torch.sort(
        torch.cat([z_vals_coarse, z_samples], dim=-1), dim=-1
    )

    # Compute 3D positions for all (N_samples + N_importance) points
    pts_fine = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_fine[..., :, None]

    # --- Stage 3: Fine rendering ---
    raw_fine = _run_network(
        pts_fine, rays_d, fine_model, embed_pos_fn, embed_dir_fn, chunk
    )

    fine_outputs = raw2outputs(raw_fine, z_vals_fine, rays_d, white_bkgd)

    # Pack both coarse and fine results
    result = {
        "rgb_map": fine_outputs["rgb_map"],
        "depth_map": fine_outputs["depth_map"],
        "acc_map": fine_outputs["acc_map"],
        "rgb_map_coarse": coarse_outputs["rgb_map"],
    }
    return result


def _run_network(pts, rays_d, model, embed_pos_fn, embed_dir_fn, chunk):
    """Query the NeRF MLP for a batch of 3D points, processing in chunks.

    Args:
        pts: Sample points, shape [N_rays, N_samples, 3].
        rays_d: Ray directions, shape [N_rays, 3].
        model: NeRF MLP module.
        embed_pos_fn: Positional encoding for xyz.
        embed_dir_fn: Positional encoding for direction.
        chunk: Max points per forward pass (prevents GPU OOM).

    Returns:
        raw: MLP outputs, shape [N_rays, N_samples, 4].
    """
    N_rays, N_samples, _ = pts.shape

    # Flatten points for batch processing
    pts_flat = pts.reshape(-1, 3)  # [N_rays * N_samples, 3]

    # Expand ray directions to match sample count, then flatten
    dirs = rays_d[:, None, :].expand_as(pts)
    dirs_flat = dirs.reshape(-1, 3)
    dirs_flat = F.normalize(dirs_flat, dim=-1)

    # Encode inputs
    pos_enc = embed_pos_fn(pts_flat)
    dir_enc = embed_dir_fn(dirs_flat)

    # Process in chunks to avoid GPU memory overflow
    outputs = []
    for i in range(0, pos_enc.shape[0], chunk):
        outputs.append(model(pos_enc[i : i + chunk], dir_enc[i : i + chunk]))
    raw = torch.cat(outputs, dim=0)

    return raw.reshape(N_rays, N_samples, 4)
