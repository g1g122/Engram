"""
Stage 2: Shuffling / SwapOpt for Position Pre-optimization.

Responsibility:
    - CORE MECHANISM: "SwapOpt" (Greedy Swapping).
    - Input:
        1. Pretrained Stage 1 model.
        2. Set of Sine Grating images (simulating retinal waves).
    - Process:
        1. Extract feature responses for all layers.
        2. Iteratively swap unit positions on the Cortical Sheet.
        3. Objective: Maximize local correlation (Minimize SL_Rel) *without* changing model weights.
    - Output: Saved permutation indices for each layer.

Algorithm (from paper lines 1579-1582):
    1) Select a cortical neighborhood at random.
    2) Compute the pairwise response correlations of all units in the neighborhood.
    3) Choose a random pair of units, and swap their locations.
    4) If swapping decreases local correlations (increases SL), undo the swap.
    5) Repeat steps 3-4 500 times.
    6) Repeat steps 1-5 10,000 times.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.backbone import ResNet18Backbone
from models.cortical_sheet import CorticalSheet, get_feature_map_sizes
from data.gratings import create_grating_dataset
from objectives.spatial_loss import (
    compute_pairwise_response_correlations,
    compute_pairwise_inverse_distances,
    pearson_correlation
)


def extract_all_responses(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Extract responses from all layers for all grating stimuli.

    Args:
        model: Pretrained ResNet18Backbone.
        dataloader: DataLoader with grating images.
        device: Device to run on.

    Returns:
        Dict mapping layer_id to response tensor of shape (num_images, num_units).
    """
    model.eval()
    all_responses = {f'L{i}': [] for i in range(2, 10)}

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc='Extracting responses'):
            images = images.to(device)
            features = model(images)

            for layer_id in all_responses.keys():
                if layer_id in features:
                    # Flatten: (B, C, H, W) -> (B, C*H*W)
                    feat = features[layer_id].view(features[layer_id].size(0), -1)
                    all_responses[layer_id].append(feat.cpu())

    # Concatenate all batches: (num_images, num_units)
    for layer_id in all_responses:
        if all_responses[layer_id]:
            all_responses[layer_id] = torch.cat(all_responses[layer_id], dim=0)

    return all_responses


def compute_neighborhood_sl(
    responses: torch.Tensor,
    positions: torch.Tensor,
    unit_indices: torch.Tensor,
    perm: torch.Tensor
) -> float:
    """
    Compute SL_Rel for a specific neighborhood.

    Args:
        responses: All responses, shape (num_images, num_units).
        positions: Base positions, shape (num_units, 2).
        unit_indices: Indices of units in neighborhood.
        perm: Current permutation.

    Returns:
        Spatial loss value (lower is better).
    """
    # Get permuted positions for these units
    permuted_positions = positions[perm[unit_indices]]

    # Compute pairwise response correlations
    response_corrs = compute_pairwise_response_correlations(
        responses.T,  # Transpose to (num_images, num_units) for correlation
        unit_indices
    )

    # Compute pairwise inverse distances
    # Create a temporary tensor with the right indices
    n = len(unit_indices)
    diff = permuted_positions.unsqueeze(0) - permuted_positions.unsqueeze(1)
    distances = torch.norm(diff, dim=2)
    inv_distances = 1.0 / (distances + 1.0)

    # Extract upper triangle
    triu_indices = torch.triu_indices(n, n, offset=1)
    pairwise_inv_dist = inv_distances[triu_indices[0], triu_indices[1]]

    # SL_Rel = 1 - Corr(response_corrs, inv_distances)
    corr = pearson_correlation(response_corrs, pairwise_inv_dist)
    return 1.0 - corr.item()


def shuffle_layer(
    responses: torch.Tensor,
    cortical_sheet: CorticalSheet,
    layer_id: str,
    num_outer_iterations: int = 10000,
    num_inner_iterations: int = 500,
    neighborhood_max_units: int = 500
) -> torch.Tensor:
    """
    Run shuffling optimization for a single layer.

    Args:
        responses: Responses for this layer, shape (num_images, num_units).
        cortical_sheet: CorticalSheet with geometry info.
        layer_id: Layer identifier.
        num_outer_iterations: Number of neighborhoods to sample.
        num_inner_iterations: Number of swaps to try per neighborhood.
        neighborhood_max_units: Max units per neighborhood.

    Returns:
        Optimized permutation tensor.
    """
    geom = cortical_sheet.geometries[layer_id]
    num_units = geom.num_units

    # Initialize identity permutation
    perm = torch.arange(num_units, dtype=torch.long)

    # Get base positions
    base_pos = getattr(cortical_sheet, f'pos_{layer_id}').clone()

    # Transpose responses for correlation computation
    responses_t = responses.T  # (num_units, num_images)

    print(f"\nShuffling layer {layer_id} ({num_units} units)")
    print(f"Outer iterations: {num_outer_iterations}, Inner iterations: {num_inner_iterations}")

    outer_pbar = tqdm(range(num_outer_iterations), desc=f'Shuffling {layer_id}')

    for outer_iter in outer_pbar:
        # Step 1: Select a random neighborhood
        center_idx = torch.randint(0, num_units, (1,)).item()
        center_pos = base_pos[perm[center_idx]]
        radius = geom.neighborhood_sigma_mm

        # Find units within neighborhood
        current_positions = base_pos[perm]
        distances = torch.norm(current_positions - center_pos, dim=1)
        within_radius = distances <= radius
        neighbor_indices = torch.where(within_radius)[0]

        # Subsample if too many
        if len(neighbor_indices) > neighborhood_max_units:
            sub_perm = torch.randperm(len(neighbor_indices))[:neighborhood_max_units]
            neighbor_indices = neighbor_indices[sub_perm]

        if len(neighbor_indices) < 3:
            continue  # Need at least 3 units

        n = len(neighbor_indices)

        # ===== OPTIMIZATION: Compute response correlation matrix ONCE =====
        # Response correlations don't change during inner loop (only positions do)
        unit_responses = responses_t[neighbor_indices]  # (n, num_images)
        centered = unit_responses - unit_responses.mean(dim=1, keepdim=True)
        norms = torch.sqrt((centered ** 2).sum(dim=1, keepdim=True) + 1e-8)
        normalized = centered / norms
        corr_matrix = normalized @ normalized.T

        # Extract upper triangle indices (reuse for all inner iterations)
        triu_idx = torch.triu_indices(n, n, offset=1)
        response_corrs = corr_matrix[triu_idx[0], triu_idx[1]]

        # Helper function to compute SL from positions only
        def compute_sl_from_positions(local_perm_indices):
            """Compute SL given local permutation indices (which positions these units occupy)."""
            pos = base_pos[local_perm_indices]
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)
            dists = torch.norm(diff, dim=2)
            inv_dists = 1.0 / (dists + 1.0)
            inv_dist_vec = inv_dists[triu_idx[0], triu_idx[1]]
            corr = pearson_correlation(response_corrs, inv_dist_vec)
            return 1.0 - corr.item()

        # Current position indices for units in this neighborhood
        local_perm = perm[neighbor_indices].clone()
        current_sl = compute_sl_from_positions(local_perm)

        # Step 3-5: Try random swaps
        num_accepted = 0
        for _ in range(num_inner_iterations):
            # Choose random pair within neighborhood (local indices)
            idx_pair = torch.randperm(n)[:2]
            local_i, local_j = idx_pair[0].item(), idx_pair[1].item()

            # Swap in local permutation
            local_perm[local_i], local_perm[local_j] = local_perm[local_j].clone(), local_perm[local_i].clone()

            # Compute new SL (only recompute distances, not response correlations)
            new_sl = compute_sl_from_positions(local_perm)

            # If SL increased (worse), undo swap
            if new_sl > current_sl:
                local_perm[local_i], local_perm[local_j] = local_perm[local_j].clone(), local_perm[local_i].clone()
            else:
                current_sl = new_sl
                num_accepted += 1

        # Apply accepted swaps back to global permutation
        perm[neighbor_indices] = local_perm

        outer_pbar.set_postfix({
            'sl': f'{current_sl:.4f}',
            'accepted': num_accepted
        })

    return perm


def compute_local_sl(
    responses_t: torch.Tensor,
    base_pos: torch.Tensor,
    unit_indices: torch.Tensor,
    perm: torch.Tensor
) -> float:
    """
    Compute local SL for a neighborhood.

    Args:
        responses_t: Transposed responses (num_units, num_images).
        base_pos: Base positions.
        unit_indices: Indices of units in neighborhood.
        perm: Current permutation.

    Returns:
        SL value.
    """
    n = len(unit_indices)
    if n < 2:
        return 0.0

    # Get responses for these units
    unit_responses = responses_t[unit_indices]  # (n, num_images)

    # Compute pairwise response correlations
    centered = unit_responses - unit_responses.mean(dim=1, keepdim=True)
    norms = torch.sqrt((centered ** 2).sum(dim=1, keepdim=True) + 1e-8)
    normalized = centered / norms
    corr_matrix = normalized @ normalized.T

    # Get permuted positions
    permuted_pos = base_pos[perm[unit_indices]]

    # Compute pairwise distances
    diff = permuted_pos.unsqueeze(0) - permuted_pos.unsqueeze(1)
    distances = torch.norm(diff, dim=2)
    inv_distances = 1.0 / (distances + 1.0)

    # Extract upper triangles
    triu_idx = torch.triu_indices(n, n, offset=1)
    response_corrs = corr_matrix[triu_idx[0], triu_idx[1]]
    inv_dist = inv_distances[triu_idx[0], triu_idx[1]]

    # SL_Rel = 1 - Corr(response_corrs, inv_distances)
    corr = pearson_correlation(response_corrs, inv_dist)
    return 1.0 - corr.item()


def run_stage2(
    checkpoint_path: str,
    config: dict,
    output_dir: str,
    device: Optional[torch.device] = None
) -> str:
    """
    Run Stage 2 shuffling.

    Args:
        checkpoint_path: Path to Stage 1 checkpoint.
        config: Configuration dictionary.
        output_dir: Directory to save permutations.
        device: Device to run on.

    Returns:
        Path to saved permutations.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained model
    print(f"Loading Stage 1 checkpoint from {checkpoint_path}")
    model = ResNet18Backbone(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Create cortical sheet
    layer_configs = config['model']['layers']
    feature_map_sizes = get_feature_map_sizes()
    cortical_sheet = CorticalSheet(layer_configs, feature_map_sizes)

    # Create grating dataset
    shuffling_config = config.get('shuffling', {})
    gratings_config = shuffling_config.get('gratings', {})
    grating_dataset = create_grating_dataset(
        img_size=gratings_config.get('img_size', 224),
        num_orientations=gratings_config.get('orientations', 8),
        num_spatial_freqs=gratings_config.get('spatial_frequencies', 8),
        num_phases=gratings_config.get('phases', 5)
    )
    grating_loader = DataLoader(grating_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Extract responses
    print("Extracting responses to gratings...")
    all_responses = extract_all_responses(model, grating_loader, device)

    # Shuffling parameters
    num_outer = shuffling_config.get('iterations', 10000)
    num_inner = shuffling_config.get('sub_iterations', 500)

    # Run shuffling for each layer
    permutations = {}
    for layer_id in cortical_sheet.geometries.keys():
        if layer_id not in all_responses or len(all_responses[layer_id]) == 0:
            continue

        responses = all_responses[layer_id]
        perm = shuffle_layer(
            responses=responses,
            cortical_sheet=cortical_sheet,
            layer_id=layer_id,
            num_outer_iterations=num_outer,
            num_inner_iterations=num_inner
        )
        permutations[layer_id] = perm
        cortical_sheet.set_permutation(layer_id, perm)

    # Save permutations
    perm_path = output_dir / 'permutations.pt'
    torch.save(permutations, perm_path)

    # Save cortical sheet with permutations
    sheet_path = output_dir / 'cortical_sheet.pt'
    cortical_sheet.save_positions(sheet_path)

    print(f"\nStage 2 complete.")
    print(f"Permutations saved to {perm_path}")
    print(f"Cortical sheet saved to {sheet_path}")

    return str(perm_path)


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Shuffling / SwapOpt')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to Stage 1 checkpoint')
    parser.add_argument('--output-dir', type=str, default='outputs/stage2',
                        help='Output directory for permutations')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Run Stage 2
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    run_stage2(
        checkpoint_path=args.checkpoint,
        config=config,
        output_dir=args.output_dir,
        device=device
    )


if __name__ == '__main__':
    main()
