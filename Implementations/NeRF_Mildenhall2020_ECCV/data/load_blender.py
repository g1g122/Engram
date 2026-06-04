"""
Blender (NeRF Synthetic) Dataset Loader.

Loads images, camera poses, and intrinsics from the NeRF Synthetic dataset
format (e.g., Lego, Chair, Drums). Each scene contains transforms_train.json,
transforms_val.json, and transforms_test.json, storing per-frame camera-to-world
matrices and image paths.

Coordinate Convention:
    Blender / OpenGL: +X Right, +Y Up, -Z Forward (camera looks along -Z).
    This is the native convention of the dataset and is preserved as-is.
"""

import json
import os

import imageio.v3 as iio
import numpy as np


def load_blender_data(basedir, factor=1, testskip=1):
    """Load a scene from the NeRF Synthetic (Blender) dataset.

    Args:
        basedir: Root directory of the scene (e.g., './nerf_synthetic/lego').
        factor: Downsample factor (1 = original resolution, 2 = half, etc.).
        testskip: Stride for loading val/test images (1 = load all).

    Returns:
        images:       np.ndarray, shape [N, H, W, 4], float32, range [0, 1]. RGBA.
        poses:        np.ndarray, shape [N, 4, 4], float32. Camera-to-world matrices.
        render_poses: np.ndarray, shape [40, 4, 4], float32. Spiral poses for novel view synthesis.
        hwf:          list [H, W, focal]. Image dimensions and focal length.
        i_split:      list of 3 np.ndarrays, indices for [train, val, test] splits.
    """
    splits = ["train", "val", "test"]
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as f:
            metas[s] = json.load(f)

    all_imgs = []
    all_poses = []
    counts = [0]

    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []

        # Subsample val/test sets to speed up evaluation during development
        skip = 1 if s == "train" else testskip

        for frame in meta["frames"][::skip]:
            fname = os.path.join(basedir, frame["file_path"] + ".png")
            imgs.append(iio.imread(fname))
            poses.append(np.array(frame["transform_matrix"]))

        # Normalize pixel values to [0, 1] and keep RGBA channels
        imgs = np.array(imgs, dtype=np.float32) / 255.0
        poses = np.array(poses, dtype=np.float32)

        all_imgs.append(imgs)
        all_poses.append(poses)
        counts.append(counts[-1] + imgs.shape[0])

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    images = np.concatenate(all_imgs, axis=0)
    poses = np.concatenate(all_poses, axis=0)

    H, W = images.shape[1], images.shape[2]
    camera_angle_x = float(metas["train"]["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    # Generate 40 render poses along a smooth circular trajectory
    render_poses = _generate_render_poses(40, angle_range=(-30, -30))

    if factor > 1:
        H = H // factor
        W = W // factor
        focal = focal / float(factor)

        imgs_down = np.zeros((images.shape[0], H, W, 4), dtype=np.float32)
        for i, img in enumerate(images):
            imgs_down[i] = _downsample_image(img, H, W)
        images = imgs_down

    return images, poses, render_poses, [H, W, focal], i_split


def _generate_render_poses(n_frames, angle_range=(-30, -30)):
    """Generate camera poses along a smooth spherical path for rendering.

    Creates a circular orbit at a fixed elevation angle, commonly used
    to produce the novel-view synthesis videos shown in the paper.

    Args:
        n_frames: Number of frames to generate.
        angle_range: Tuple (theta, phi_offset) controlling elevation.

    Returns:
        np.ndarray, shape [n_frames, 4, 4], float32.
    """
    render_poses = []
    for theta in np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False):
        c2w = _pose_spherical(theta, angle_range[0], 4.0)
        render_poses.append(c2w)
    return np.array(render_poses, dtype=np.float32)


def _pose_spherical(theta, phi, radius):
    """Construct a camera-to-world matrix from spherical coordinates.

    Places the camera at (radius, theta, phi) in spherical coordinates,
    looking toward the origin. Uses Blender/OpenGL convention.

    Args:
        theta: Azimuth angle in radians.
        phi:   Elevation angle in degrees.
        radius: Distance from origin.

    Returns:
        np.ndarray, shape [4, 4], float32. Camera-to-world matrix.
    """
    phi_rad = np.deg2rad(phi)

    # Translation: camera at distance `radius` along -Z, then rotated
    c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, radius],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    # Rotate around X axis by -phi (elevation)
    rot_phi = np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi_rad), -np.sin(phi_rad), 0],
        [0, np.sin(phi_rad), np.cos(phi_rad), 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    # Rotate around Y axis by theta (azimuth)
    rot_theta = np.array([
        [np.cos(theta), 0, -np.sin(theta), 0],
        [0, 1, 0, 0],
        [np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    # Blender coordinate fix: swap Y and Z axes to convert
    # from mathematical spherical to Blender/OpenGL convention
    blender_fix = np.array([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    c2w = rot_theta @ rot_phi @ c2w
    c2w = blender_fix @ c2w
    return c2w


def _downsample_image(img, target_h, target_w):
    """Downsample an image using simple 2x2 area averaging.

    Args:
        img: np.ndarray, shape [H, W, C], float32.
        target_h: Target height (H // 2).
        target_w: Target width (W // 2).

    Returns:
        np.ndarray, shape [target_h, target_w, C], float32.
    """
    # Reshape into 2x2 blocks and average
    h, w, c = img.shape
    return img.reshape(target_h, 2, target_w, 2, c).mean(axis=(1, 3))
