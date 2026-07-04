"""
VTC (Ventral Temporal Cortex) Metrics for TDANN.

Responsibility:
    - Quantify VTC Topography using PHYSICAL cortical coordinates.
    - Metrics (following paper methods):
        1. Selectivity Maps: t-statistic for each category.
        2. Patch Detection: smoothed, thresholded, size-filtered.
        3. RSM: Representational Similarity Matrix.
        4. Overlap Analysis: correlation of category proportions.
        
Paper parameters for TDANN (line 1820):
    - Selectivity threshold: t > 2
    - Smoothing σ: 2.4 mm
    - Min patch size: 100 mm²
    - Max patch size: 4,500 mm²
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import ttest_ind
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


# Paper-defined parameters for TDANN
TDANN_SELECTIVITY_THRESHOLD = 2.0
TDANN_SMOOTHING_SIGMA = 2.4  # mm
TDANN_MIN_PATCH_SIZE_MM2 = 100
TDANN_MAX_PATCH_SIZE_MM2 = 4500


def compute_category_responses_with_positions(
    model: nn.Module,
    cortical_sheet,
    floc_loader: DataLoader,
    vtc_layer_id: str = 'L9',
    device: torch.device = None
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Extract VTC layer responses with physical positions."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    category_responses = {}

    with torch.no_grad():
        for images, cat_idxs, cat_names, _ in floc_loader:
            images = images.to(device)
            features = model(images)

            if vtc_layer_id in features:
                feat = features[vtc_layer_id]
                feat = feat.view(feat.size(0), -1).cpu()

                for i, cat_name in enumerate(cat_names):
                    if cat_name not in category_responses:
                        category_responses[cat_name] = []
                    category_responses[cat_name].append(feat[i])

    for cat_name in category_responses:
        category_responses[cat_name] = torch.stack(category_responses[cat_name])

    positions = cortical_sheet.get_positions(vtc_layer_id).cpu()

    return category_responses, positions


# ============================================================================
# Selectivity Computation
# ============================================================================

def compute_selectivity_tstat(
    category_responses: Dict[str, torch.Tensor],
    target_category: str
) -> torch.Tensor:
    """
    Compute t-statistic for selectivity to a target category.
    
    Paper reference (Equation 8, lines 1734-1745):
    t = (μ_on - μ_off) / sqrt(σ²_on/N_on + σ²_off/N_off)
    
    Returns:
        Tensor of shape (num_units,) with t-statistics.
    """
    target_resp = category_responses[target_category].numpy()
    
    other_resps = []
    for cat, resp in category_responses.items():
        if cat != target_category:
            other_resps.append(resp.numpy())
    other_resp = np.concatenate(other_resps, axis=0)
    
    num_units = target_resp.shape[1]
    t_stats = np.zeros(num_units)
    
    for i in range(num_units):
        on_responses = target_resp[:, i]
        off_responses = other_resp[:, i]
        
        # Welch's t-test
        n_on, n_off = len(on_responses), len(off_responses)
        mean_on, mean_off = on_responses.mean(), off_responses.mean()
        var_on, var_off = on_responses.var(ddof=1), off_responses.var(ddof=1)
        
        se = np.sqrt(var_on / n_on + var_off / n_off + 1e-8)
        t_stats[i] = (mean_on - mean_off) / se
    
    return torch.from_numpy(t_stats).float()


def compute_all_selectivity_maps(
    category_responses: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Compute selectivity t-statistics for all categories."""
    selectivity_maps = {}
    for cat in category_responses.keys():
        t_stats = compute_selectivity_tstat(category_responses, cat)
        selectivity_maps[cat] = t_stats
    return selectivity_maps


# ============================================================================
# Selectivity Map Smoothing and Interpolation
# ============================================================================

def smooth_selectivity_map(
    positions: torch.Tensor,
    selectivity: torch.Tensor,
    sigma_mm: float = TDANN_SMOOTHING_SIGMA,
    grid_resolution: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth and interpolate selectivity map to regular grid.
    
    Paper reference (line 1797): "The first step in identifying patches 
    is to smooth and interpolate discrete selectivity maps."
    
    Returns:
        Tuple of (grid_x, grid_y, smoothed_selectivity).
    """
    pos_np = positions.numpy()
    sel_np = selectivity.numpy()
    
    x_min, x_max = pos_np[:, 0].min(), pos_np[:, 0].max()
    y_min, y_max = pos_np[:, 1].min(), pos_np[:, 1].max()
    
    # Interpolate to grid
    grid_x, grid_y = np.mgrid[x_min:x_max:complex(grid_resolution),
                               y_min:y_max:complex(grid_resolution)]
    
    grid_sel = griddata(pos_np, sel_np, (grid_x, grid_y), method='linear')
    
    # Fill NaN with nearest neighbor
    nan_mask = np.isnan(grid_sel)
    if nan_mask.any():
        grid_sel_nearest = griddata(pos_np, sel_np, (grid_x, grid_y), method='nearest')
        grid_sel[nan_mask] = grid_sel_nearest[nan_mask]
    
    # Gaussian smoothing
    # Convert sigma from mm to pixels
    pixel_size = (x_max - x_min) / grid_resolution
    sigma_pixels = sigma_mm / pixel_size
    
    smoothed_sel = gaussian_filter(grid_sel, sigma=sigma_pixels)
    
    return grid_x, grid_y, smoothed_sel


# ============================================================================
# Patch Detection
# ============================================================================

def detect_patches_on_grid(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_selectivity: np.ndarray,
    threshold: float = TDANN_SELECTIVITY_THRESHOLD,
    min_size_mm2: float = TDANN_MIN_PATCH_SIZE_MM2,
    max_size_mm2: float = TDANN_MAX_PATCH_SIZE_MM2
) -> List[Dict]:
    """
    Detect patches in smoothed selectivity map.
    
    Paper reference (lines 1795-1801):
    1. Threshold selectivity map
    2. Find contiguous regions
    3. Filter by size (100 - 4500 mm²)
    
    Returns:
        List of patch dicts with centroid, size, peak_t.
    """
    from scipy import ndimage
    
    # Threshold
    binary_map = grid_selectivity > threshold
    
    # Label connected components
    labeled, num_features = ndimage.label(binary_map)
    
    # Compute pixel area
    h, w = grid_x.shape
    x_range = grid_x.max() - grid_x.min()
    y_range = grid_y.max() - grid_y.min()
    pixel_area_mm2 = (x_range / (h - 1)) * (y_range / (w - 1))
    
    patches = []
    for label_id in range(1, num_features + 1):
        mask = labeled == label_id
        pixel_count = mask.sum()
        area_mm2 = pixel_count * pixel_area_mm2
        
        # Filter by size
        if area_mm2 < min_size_mm2 or area_mm2 > max_size_mm2:
            continue
        
        # Compute properties
        patch_x = grid_x[mask]
        patch_y = grid_y[mask]
        patch_sel = grid_selectivity[mask]
        
        patches.append({
            'centroid': (patch_x.mean(), patch_y.mean()),
            'size_mm2': area_mm2,
            'peak_t': patch_sel.max(),
            'mean_t': patch_sel.mean(),
            'num_pixels': pixel_count
        })
    
    return patches


def detect_category_patches_physical(
    positions: torch.Tensor,
    selectivity: torch.Tensor,
    threshold: float = TDANN_SELECTIVITY_THRESHOLD,
    sigma_mm: float = TDANN_SMOOTHING_SIGMA,
    min_size_mm2: float = TDANN_MIN_PATCH_SIZE_MM2,
    max_size_mm2: float = TDANN_MAX_PATCH_SIZE_MM2
) -> List[Dict]:
    """
    Detect category-selective patches.
    
    Full pipeline: smooth → threshold → find connected regions → filter by size.
    """
    grid_x, grid_y, smoothed_sel = smooth_selectivity_map(
        positions, selectivity, sigma_mm
    )
    
    patches = detect_patches_on_grid(
        grid_x, grid_y, smoothed_sel,
        threshold, min_size_mm2, max_size_mm2
    )
    
    return patches


# ============================================================================
# Overlap Analysis
# ============================================================================

def compute_category_overlap(
    positions: torch.Tensor,
    selectivity_x: torch.Tensor,
    selectivity_y: torch.Tensor,
    neighborhood_size_mm: float = 10.0,
    selectivity_threshold: float = 4.0
) -> float:
    """
    Compute overlap between two category-selective populations.
    
    Paper reference (lines 1804-1813, Equation 9):
    1. Bin cortical sheet into 10mm x 10mm neighborhoods
    2. Count fraction of selective units for each category per neighborhood
    3. Overlap = (1 - SpearmanCorr(X, Y)) / 2
    
    Returns:
        Overlap score in [0, 1]. Lower = more overlap.
    """
    from scipy.stats import spearmanr
    
    pos_np = positions.numpy()
    sel_x = selectivity_x.numpy()
    sel_y = selectivity_y.numpy()
    
    # Create bins
    x_min, x_max = pos_np[:, 0].min(), pos_np[:, 0].max()
    y_min, y_max = pos_np[:, 1].min(), pos_np[:, 1].max()
    
    x_bins = np.arange(x_min, x_max + neighborhood_size_mm, neighborhood_size_mm)
    y_bins = np.arange(y_min, y_max + neighborhood_size_mm, neighborhood_size_mm)
    
    frac_x = []
    frac_y = []
    
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            # Find units in this neighborhood
            in_bin = (
                (pos_np[:, 0] >= x_bins[i]) & (pos_np[:, 0] < x_bins[i + 1]) &
                (pos_np[:, 1] >= y_bins[j]) & (pos_np[:, 1] < y_bins[j + 1])
            )
            
            if in_bin.sum() == 0:
                continue
            
            # Fraction selective for each category
            fx = (sel_x[in_bin] > selectivity_threshold).mean()
            fy = (sel_y[in_bin] > selectivity_threshold).mean()
            
            frac_x.append(fx)
            frac_y.append(fy)
    
    if len(frac_x) < 3:
        return 0.5  # Not enough data
    
    corr, _ = spearmanr(frac_x, frac_y)
    if np.isnan(corr):
        corr = 0.0
    
    overlap = (1 - corr) / 2
    return overlap


# ============================================================================
# RSM
# ============================================================================

def compute_rsm(
    category_responses: Dict[str, torch.Tensor]
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute Representational Similarity Matrix.
    
    Paper reference (lines 1781-1785): "we compute a representational 
    similarity matrix (RSM) as the pairwise Pearson's correlation between 
    patterns of selectivity for each of the five fLoc categories."
    """
    categories = list(category_responses.keys())
    num_cats = len(categories)

    # Mean response per category
    mean_responses = []
    for cat in categories:
        mean_resp = category_responses[cat].mean(dim=0)
        mean_responses.append(mean_resp)
    mean_responses = torch.stack(mean_responses)

    # Pearson correlation
    centered = mean_responses - mean_responses.mean(dim=1, keepdim=True)
    norms = torch.sqrt((centered ** 2).sum(dim=1, keepdim=True) + 1e-8)
    normalized = centered / norms

    rsm = (normalized @ normalized.T).numpy()

    return rsm, categories


# ============================================================================
# Smoothness Profile (for VTC)
# ============================================================================

def compute_vtc_smoothness_profile(
    positions: torch.Tensor,
    selectivity: torch.Tensor,
    num_bins: int = 20,
    max_distance_mm: Optional[float] = None,
    num_samples: int = 10000,
    min_mean_response: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute selectivity difference vs cortical distance for VTC.
    
    Paper reference (lines 1787-1790): "We draw 25 random samples of 500 
    units each. Each sample is filtered to include only units with a mean 
    response of at least 0.5 a.u."
    """
    n = positions.shape[0]
    sel_np = selectivity.numpy()
    
    # Sample random pairs
    idx1 = torch.randint(0, n, (num_samples,))
    idx2 = torch.randint(0, n, (num_samples,))
    
    distances = torch.norm(positions[idx1] - positions[idx2], dim=1).numpy()
    sel_diff = np.abs(sel_np[idx1] - sel_np[idx2])
    
    if max_distance_mm is None:
        max_distance_mm = distances.max()
    
    bin_edges = np.linspace(0, max_distance_mm, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    mean_diffs = []
    for i in range(num_bins):
        mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
        if mask.sum() > 0:
            mean_diff = sel_diff[mask].mean()
        else:
            mean_diff = np.nan
        mean_diffs.append(mean_diff)
    
    mean_diffs = np.array(mean_diffs)
    
    # Normalize by chance
    chance_level = sel_diff.mean()
    mean_diffs = mean_diffs / (chance_level + 1e-8)
    
    return bin_centers, mean_diffs


# ============================================================================
# Visualization
# ============================================================================

def plot_selectivity_map_physical(
    positions: torch.Tensor,
    selectivity: torch.Tensor,
    category_name: str,
    patches: Optional[List[Dict]] = None,
    point_size: float = 1.0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot selectivity map as scatter."""
    fig, ax = plt.subplots(figsize=(10, 10))

    pos_np = positions.numpy()
    sel_np = selectivity.numpy()
    sel_clipped = np.clip(sel_np, -10, 10)

    scatter = ax.scatter(
        pos_np[:, 0], pos_np[:, 1],
        c=sel_clipped, cmap='RdBu_r', vmin=-10, vmax=10,
        s=point_size, alpha=0.8
    )

    plt.colorbar(scatter, ax=ax, label='t-statistic')

    if patches:
        for patch in patches:
            cx, cy = patch['centroid']
            ax.scatter(cx, cy, c='black', s=100, marker='*', edgecolors='white')

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'{category_name.capitalize()} Selectivity')
    ax.set_aspect('equal')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_selectivity_map_smoothed(
    positions: torch.Tensor,
    selectivity: torch.Tensor,
    category_name: str,
    sigma_mm: float = TDANN_SMOOTHING_SIGMA,
    threshold: float = TDANN_SELECTIVITY_THRESHOLD,
    patches: Optional[List[Dict]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot smoothed and thresholded selectivity map (paper style)."""
    grid_x, grid_y, smoothed_sel = smooth_selectivity_map(
        positions, selectivity, sigma_mm
    )
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    x_min, x_max = grid_x.min(), grid_x.max()
    y_min, y_max = grid_y.min(), grid_y.max()
    extent = [x_min, x_max, y_min, y_max]
    
    # Clip for visualization
    sel_clipped = np.clip(smoothed_sel, -10, 10)
    
    im = ax.imshow(sel_clipped.T, origin='lower', extent=extent,
                   cmap='RdBu_r', vmin=-10, vmax=10, aspect='equal')
    
    # Contour for threshold
    ax.contour(grid_x, grid_y, smoothed_sel, levels=[threshold],
               colors='black', linewidths=1)
    
    plt.colorbar(im, ax=ax, label='t-statistic')
    
    if patches:
        for patch in patches:
            cx, cy = patch['centroid']
            ax.scatter(cx, cy, c='white', s=100, marker='*', edgecolors='black')

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'{category_name.capitalize()} Selectivity (smoothed)')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_all_selectivity_maps_physical(
    positions: torch.Tensor,
    selectivity_maps: Dict[str, torch.Tensor],
    point_size: float = 0.5,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot selectivity maps for all categories."""
    categories = list(selectivity_maps.keys())
    n_cats = len(categories)
    n_cols = min(3, n_cats)
    n_rows = (n_cats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    pos_np = positions.numpy()

    for idx, cat in enumerate(categories):
        ax = axes[idx]
        sel_np = selectivity_maps[cat].numpy()
        sel_clipped = np.clip(sel_np, -10, 10)

        scatter = ax.scatter(
            pos_np[:, 0], pos_np[:, 1],
            c=sel_clipped, cmap='RdBu_r', vmin=-10, vmax=10,
            s=point_size, alpha=0.8
        )

        ax.set_title(f'{cat.capitalize()}')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n_cats, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_rsm(
    rsm: np.ndarray,
    categories: List[str],
    title: str = 'Representational Similarity Matrix',
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot RSM."""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(rsm, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Correlation')

    ax.set_xticks(range(len(categories)))
    ax.set_yticks(range(len(categories)))
    ax.set_xticklabels([c.capitalize() for c in categories], rotation=45, ha='right')
    ax.set_yticklabels([c.capitalize() for c in categories])

    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
