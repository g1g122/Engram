"""
V1 Metrics for TDANN.

Responsibility:
    - Quantify V1 Topography using PHYSICAL cortical coordinates.
    - Metrics (following paper methods):
        1. Orientation Tuning: von Mises fit or vector sum for preferred orientation.
        2. OPM Grid Interpolation: circular mean of nearby units.
        3. Pinwheel Detection: winding number on interpolated grid.
        4. Smoothness Profile: orientation diff vs distance, normalized by chance.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from scipy.optimize import curve_fit


# ============================================================================
# Orientation Tuning Functions
# ============================================================================

def von_mises_function(theta: np.ndarray, A: float, kappa: float, theta0: float, baseline: float) -> np.ndarray:
    """
    von Mises function for orientation tuning.
    
    f(θ) = A * exp(κ * cos(2*(θ - θ₀))) + baseline
    
    The factor of 2 accounts for 180° periodicity of orientation.
    """
    return A * np.exp(kappa * np.cos(2 * (theta - theta0))) + baseline


def fit_von_mises(orientations: np.ndarray, responses: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit von Mises function to get preferred orientation.
    
    Args:
        orientations: Array of orientations in radians.
        responses: Array of response magnitudes.
    
    Returns:
        Tuple of (preferred_orientation_rad, kappa, peak_response).
    """
    try:
        # Initial guess
        peak_idx = np.argmax(responses)
        theta0_init = orientations[peak_idx]
        A_init = responses.max() - responses.min()
        baseline_init = responses.min()
        
        # Fit
        popt, _ = curve_fit(
            von_mises_function, orientations, responses,
            p0=[A_init, 2.0, theta0_init, baseline_init],
            bounds=([0, 0, -np.pi, 0], [np.inf, 100, np.pi, np.inf]),
            maxfev=1000
        )
        
        A, kappa, theta0, baseline = popt
        return theta0, kappa, A + baseline
        
    except Exception:
        # Fallback to vector sum
        peak_idx = np.argmax(responses)
        return orientations[peak_idx], 1.0, responses.max()


def compute_orientation_tuning_detailed(
    responses: torch.Tensor,
    orientations: List[float],
    use_von_mises: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute orientation tuning properties for each unit.

    Args:
        responses: Tensor of shape (num_orientations, num_units).
        orientations: List of orientations in degrees.
        use_von_mises: If True, fit von Mises. Otherwise, use vector sum.

    Returns:
        Tuple of:
            - preferred_ori: Preferred orientations in degrees [0, 180).
            - circular_variance: CV values in [0, 1] (lower = sharper tuning).
            - tuning_magnitude: Peak-to-peak tuning strength.
    """
    num_units = responses.shape[1]
    num_orientations = len(orientations)
    
    responses_np = responses.numpy()
    orientations_rad = np.array([math.radians(o) for o in orientations])
    
    # Use numpy arrays for computation
    preferred_ori = np.zeros(num_units, dtype=np.float32)
    circular_variance = np.zeros(num_units, dtype=np.float32)
    tuning_magnitude = np.zeros(num_units, dtype=np.float32)
    
    for i in range(num_units):
        unit_responses = responses_np[:, i]
        
        # Tuning magnitude = peak - trough
        tuning_magnitude[i] = float(unit_responses.max() - unit_responses.min())
        
        if use_von_mises:
            theta0, kappa, _ = fit_von_mises(orientations_rad, unit_responses)
            preferred_ori[i] = float(math.degrees(theta0) % 180)
            # CV from kappa: CV ≈ exp(-kappa) for large kappa
            circular_variance[i] = float(np.exp(-kappa))
        else:
            # Vector sum method
            theta_2x = np.array([math.radians(o * 2) for o in orientations])
            cos_sum = (unit_responses * np.cos(theta_2x)).sum()
            sin_sum = (unit_responses * np.sin(theta_2x)).sum()
            response_sum = unit_responses.sum()
            
            # Preferred orientation
            pref_rad = np.arctan2(sin_sum, cos_sum) / 2
            preferred_ori[i] = float(math.degrees(pref_rad) % 180)
            
            # Circular variance
            vector_length = np.sqrt(cos_sum**2 + sin_sum**2)
            cv = 1 - vector_length / (response_sum + 1e-8)
            circular_variance[i] = float(np.clip(cv, 0, 1))
    
    # Convert to torch tensors
    return (torch.from_numpy(preferred_ori), 
            torch.from_numpy(circular_variance), 
            torch.from_numpy(tuning_magnitude))


def filter_by_tuning_strength(
    preferred_ori: torch.Tensor,
    circular_variance: torch.Tensor,
    tuning_magnitude: torch.Tensor,
    positions: torch.Tensor,
    top_percentile: float = 25.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter to keep only top N% of units by tuning magnitude.
    
    Paper reference (line 1706): "restrict to those with the highest 25% 
    peak-to-peak tuning curve magnitudes"
    
    Args:
        preferred_ori: Preferred orientations (num_units,).
        circular_variance: CV values (num_units,).
        tuning_magnitude: Tuning strength (num_units,).
        positions: Physical positions (num_units, 2).
        top_percentile: Keep top N% of units.
    
    Returns:
        Filtered versions of all inputs.
    """
    threshold = np.percentile(tuning_magnitude.numpy(), 100 - top_percentile)
    mask = tuning_magnitude >= threshold
    
    return (
        preferred_ori[mask],
        circular_variance[mask],
        tuning_magnitude[mask],
        positions[mask]
    )


# ============================================================================
# Response Extraction
# ============================================================================

def extract_v1_responses_with_positions(
    model: nn.Module,
    cortical_sheet,
    grating_loader: DataLoader,
    v1_layer_id: str = 'L4',
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """Extract V1 layer responses with physical positions."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    all_responses = []
    all_orientations = []

    with torch.no_grad():
        for images, params in grating_loader:
            images = images.to(device)
            features = model(images)

            if v1_layer_id in features:
                feat = features[v1_layer_id]
                feat = feat.view(feat.size(0), -1)
                all_responses.append(feat.cpu())

            if isinstance(params, dict):
                for ori in params['orientation']:
                    all_orientations.append(ori.item() if hasattr(ori, 'item') else ori)
            else:
                for p in params:
                    all_orientations.append(p['orientation'])

    responses = torch.cat(all_responses, dim=0)
    positions = cortical_sheet.get_positions(v1_layer_id).cpu()

    return responses, positions, all_orientations


def compute_opm_with_positions(
    responses: torch.Tensor,
    orientations: List[float],
    use_von_mises: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute OPM properties from responses.
    
    Returns:
        Tuple of (preferred_ori, circular_variance, tuning_magnitude).
    """
    unique_oris = sorted(set(orientations))
    ori_responses = []

    for ori in unique_oris:
        mask = [o == ori for o in orientations]
        ori_resp = responses[mask].mean(dim=0)
        ori_responses.append(ori_resp)

    ori_responses = torch.stack(ori_responses)
    return compute_orientation_tuning_detailed(ori_responses, unique_oris, use_von_mises)


# ============================================================================
# OPM Grid Interpolation
# ============================================================================

def interpolate_opm_to_grid(
    positions: torch.Tensor,
    preferred_ori: torch.Tensor,
    grid_resolution: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate OPM onto regular grid using circular mean.
    
    Paper reference (line 1690): "We interpolate the OPM onto a two-dimensional 
    grid by computing the circular mean of the preferred orientation of units 
    near a given location."
    
    Returns:
        Tuple of (grid_x, grid_y, grid_ori, reliability_mask).
    """
    pos_np = positions.numpy()
    ori_np = preferred_ori.numpy()

    x_min, x_max = pos_np[:, 0].min(), pos_np[:, 0].max()
    y_min, y_max = pos_np[:, 1].min(), pos_np[:, 1].max()

    grid_x, grid_y = np.mgrid[x_min:x_max:complex(grid_resolution),
                               y_min:y_max:complex(grid_resolution)]

    # Circular interpolation via complex exponential
    ori_rad = np.deg2rad(ori_np * 2)  # Double for 180° periodicity
    complex_ori = np.exp(1j * ori_rad)

    grid_complex = griddata(pos_np, complex_ori, (grid_x, grid_y), method='linear')
    
    # Reliability mask: unreliable if interpolation failed (NaN)
    reliability_mask = ~np.isnan(grid_complex)
    
    grid_ori = np.angle(grid_complex) / 2
    grid_ori = np.rad2deg(grid_ori) % 180

    return grid_x, grid_y, grid_ori, reliability_mask


# ============================================================================
# Pinwheel Detection
# ============================================================================

def detect_pinwheels_on_grid(
    grid_ori: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    reliability_mask: np.ndarray,
    winding_threshold: float = 0.5
) -> List[Tuple[float, float, int]]:
    """
    Detect pinwheels using winding number on interpolated grid.
    
    Paper reference (lines 1692-1698): "Each grid location is assigned a 
    'winding number', computed by considering the preferred orientations 
    of the eight pixels directly bordering the pixel under consideration."
    
    Returns:
        List of (x, y, sign) tuples. sign = +1 for CW, -1 for CCW pinwheel.
    """
    h, w = grid_ori.shape
    pinwheels = []
    
    # 8-connected neighborhood offsets (clockwise starting from top-left)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), 
               (1, 1), (1, 0), (1, -1), (0, -1)]
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # Skip unreliable pixels
            if not reliability_mask[i, j]:
                continue
            
            # Get orientations of 8 neighbors (clockwise)
            neighbor_oris = []
            all_reliable = True
            
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if not reliability_mask[ni, nj]:
                    all_reliable = False
                    break
                neighbor_oris.append(grid_ori[ni, nj])
            
            if not all_reliable:
                continue
            
            # Compute winding number
            total_change = 0
            for k in range(8):
                diff = neighbor_oris[(k + 1) % 8] - neighbor_oris[k]
                # Wrap to [-90, 90] for 180° periodicity
                while diff > 90:
                    diff -= 180
                while diff < -90:
                    diff += 180
                total_change += diff
            
            # Winding number (should be ±180 for pinwheel)
            winding = total_change / 180.0
            
            if abs(winding) > winding_threshold:
                sign = 1 if winding > 0 else -1
                pinwheels.append((grid_x[i, j], grid_y[i, j], sign))
    
    return pinwheels


def detect_pinwheels_physical(
    positions: torch.Tensor,
    preferred_ori: torch.Tensor,
    grid_resolution: int = 200
) -> List[Tuple[float, float]]:
    """Wrapper for pinwheel detection."""
    grid_x, grid_y, grid_ori, mask = interpolate_opm_to_grid(
        positions, preferred_ori, grid_resolution
    )
    pinwheels = detect_pinwheels_on_grid(grid_ori, grid_x, grid_y, mask)
    return [(x, y) for x, y, _ in pinwheels]


def compute_pinwheel_density(
    pinwheels: List[Tuple[float, float]],
    cortical_area_mm2: float
) -> float:
    """Compute pinwheel density (pinwheels per mm²)."""
    return len(pinwheels) / cortical_area_mm2


# ============================================================================
# Smoothness Analysis
# ============================================================================

def compute_smoothness_profile_physical(
    positions: torch.Tensor,
    preferred_ori: torch.Tensor,
    num_bins: int = 20,
    max_distance_mm: Optional[float] = None,
    num_samples: int = 10000,
    normalize_by_chance: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute orientation difference vs cortical distance profile.
    
    Paper reference (lines 1705-1714): "pairs of units are binned according 
    to their distance, and the average absolute difference in preferred 
    orientation is plotted for each distance bin. Finally, we divide the 
    pairwise difference by the chance value obtained by random resampling."
    
    Returns:
        Tuple of (distance_centers, normalized_ori_difference).
    """
    n = positions.shape[0]
    
    # Sample random pairs
    idx1 = torch.randint(0, n, (num_samples,))
    idx2 = torch.randint(0, n, (num_samples,))
    
    # Compute distances
    distances = torch.norm(positions[idx1] - positions[idx2], dim=1)
    
    # Compute orientation differences (circular, in [0, 90])
    ori_diff = torch.abs(preferred_ori[idx1] - preferred_ori[idx2])
    ori_diff = torch.min(ori_diff, 180 - ori_diff)
    
    if max_distance_mm is None:
        max_distance_mm = distances.max().item()
    
    bin_edges = np.linspace(0, max_distance_mm, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Compute mean difference per bin
    mean_diffs = []
    for i in range(num_bins):
        mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
        if mask.sum() > 0:
            mean_diff = ori_diff[mask].mean().item()
        else:
            mean_diff = np.nan
        mean_diffs.append(mean_diff)
    
    mean_diffs = np.array(mean_diffs)
    
    # Normalize by chance level (paper line 1713-1714)
    if normalize_by_chance:
        # Chance = average of all pairs (random shuffling)
        chance_level = ori_diff.mean().item()
        mean_diffs = mean_diffs / chance_level
    
    return bin_centers, mean_diffs


# ============================================================================
# Visualization
# ============================================================================

def plot_opm_physical(
    positions: torch.Tensor,
    preferred_ori: torch.Tensor,
    pinwheels: Optional[List[Tuple[float, float]]] = None,
    title: str = 'V1 Orientation Preference Map',
    point_size: float = 1.0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot OPM as scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 10))

    pos_np = positions.numpy()
    ori_np = preferred_ori.numpy()

    scatter = ax.scatter(
        pos_np[:, 0], pos_np[:, 1],
        c=ori_np, cmap='hsv', vmin=0, vmax=180,
        s=point_size, alpha=0.8
    )

    plt.colorbar(scatter, ax=ax, label='Preferred Orientation (°)')

    if pinwheels:
        pw_x = [p[0] for p in pinwheels]
        pw_y = [p[1] for p in pinwheels]
        ax.scatter(pw_x, pw_y, c='black', s=20, marker='x', linewidths=1)

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(title)
    ax.set_aspect('equal')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_opm_interpolated(
    positions: torch.Tensor,
    preferred_ori: torch.Tensor,
    pinwheels: Optional[List[Tuple[float, float]]] = None,
    title: str = 'V1 Orientation Preference Map',
    grid_resolution: int = 200,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot OPM using grid interpolation (paper style)."""
    grid_x, grid_y, grid_ori, mask = interpolate_opm_to_grid(
        positions, preferred_ori, grid_resolution
    )
    
    # Mask unreliable pixels
    grid_ori_masked = np.ma.masked_where(~mask, grid_ori)

    fig, ax = plt.subplots(figsize=(10, 10))

    x_min, x_max = grid_x.min(), grid_x.max()
    y_min, y_max = grid_y.min(), grid_y.max()
    extent = [x_min, x_max, y_min, y_max]
    
    im = ax.imshow(grid_ori_masked.T, origin='lower', extent=extent,
                   cmap='hsv', vmin=0, vmax=180, aspect='equal')

    plt.colorbar(im, ax=ax, label='Preferred Orientation (°)')

    if pinwheels:
        pw_x = [p[0] for p in pinwheels]
        pw_y = [p[1] for p in pinwheels]
        ax.scatter(pw_x, pw_y, s=30, marker='o', 
                   facecolors='none', edgecolors='black', linewidths=1.5)

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_smoothness_profile(
    distances: np.ndarray,
    differences: np.ndarray,
    title: str = 'Orientation Smoothness Profile',
    normalized: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot smoothness profile."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(distances, differences, 'b-o', linewidth=2, markersize=4)
    ax.axhline(y=1.0 if normalized else 45, color='gray', linestyle='--', 
               label='Chance level')
    
    ax.set_xlabel('Cortical Distance (mm)')
    if normalized:
        ax.set_ylabel('Normalized Orientation Difference')
    else:
        ax.set_ylabel('Orientation Difference (°)')
    ax.set_title(title)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
