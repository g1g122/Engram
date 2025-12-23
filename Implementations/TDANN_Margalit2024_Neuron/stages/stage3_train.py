"""
Stage 3: Final Training with Spatial + Task Loss.

Responsibility:
    - Train the final TDANN model.
    - Protocol:
        1. Load the "Optimal Permutation" from Stage 2 into the Cortical Sheet.
        2. Randomly Re-Initialize ALL model weights (Do NOT use Stage 1 weights).
        3. Train end-to-end using: Loss = Task_Loss (SimCLR) + alpha * Spatial_Loss (SL_Rel).
    - Key Parameter: alpha (Spatial Loss Weight, typically 0.25).
"""

import argparse
from pathlib import Path
from typing import Optional
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.tdann import create_tdann
from models.cortical_sheet import get_feature_map_sizes
from objectives.simclr_loss import SimCLRLoss
from data.imagenet import create_simclr_dataloader, create_standard_dataloader


def train_epoch_simclr(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    simclr_criterion: SimCLRLoss,
    device: torch.device,
    epoch: int
) -> dict:
    """
    Train one epoch with SimCLR + Spatial Loss.

    Returns dict with 'total_loss', 'task_loss', 'spatial_loss'.
    """
    model.train()

    total_loss_sum = 0.0
    task_loss_sum = 0.0
    spatial_loss_sum = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for (view1, view2), _ in pbar:
        view1 = view1.to(device)
        view2 = view2.to(device)

        optimizer.zero_grad()

        # Forward pass for both views
        features1 = model(view1, return_projections=True)
        features2 = model(view2, return_projections=True)

        # Task loss: SimCLR on projected features
        z1 = features1['projections']
        z2 = features2['projections']
        task_loss = simclr_criterion(z1, z2)

        # Spatial loss: computed on concatenated features from both views
        # Use features from view1 for spatial loss (or average both)
        spatial_loss, _ = model.compute_spatial_loss(features1)

        # Total loss
        total_loss = task_loss + spatial_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        total_loss_sum += total_loss.item()
        task_loss_sum += task_loss.item()
        spatial_loss_sum += spatial_loss.item()
        num_batches += 1

        pbar.set_postfix({
            'total': f'{total_loss.item():.4f}',
            'task': f'{task_loss.item():.4f}',
            'spatial': f'{spatial_loss.item():.4f}'
        })

    return {
        'total_loss': total_loss_sum / max(num_batches, 1),
        'task_loss': task_loss_sum / max(num_batches, 1),
        'spatial_loss': spatial_loss_sum / max(num_batches, 1)
    }


def train_epoch_supervised(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    ce_criterion: nn.CrossEntropyLoss,
    device: torch.device,
    epoch: int
) -> dict:
    """
    Train one epoch with Supervised + Spatial Loss.
    """
    model.train()

    total_loss_sum = 0.0
    task_loss_sum = 0.0
    spatial_loss_sum = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        features = model(images)

        # Task loss: Cross entropy
        logits = features['logits']
        task_loss = ce_criterion(logits, labels)

        # Spatial loss
        spatial_loss, _ = model.compute_spatial_loss(features)

        # Total loss
        total_loss = task_loss + spatial_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        total_loss_sum += total_loss.item()
        task_loss_sum += task_loss.item()
        spatial_loss_sum += spatial_loss.item()
        num_batches += 1

        pbar.set_postfix({
            'total': f'{total_loss.item():.4f}',
            'task': f'{task_loss.item():.4f}',
            'spatial': f'{spatial_loss.item():.4f}'
        })

    return {
        'total_loss': total_loss_sum / max(num_batches, 1),
        'task_loss': task_loss_sum / max(num_batches, 1),
        'spatial_loss': spatial_loss_sum / max(num_batches, 1)
    }


def run_stage3(
    hf_dataset,
    permutation_path: str,
    config: dict,
    output_dir: str,
    use_simclr: bool = True,
    device: Optional[torch.device] = None
) -> str:
    """
    Run Stage 3 training.

    Args:
        hf_dataset: HuggingFace dataset for training.
        permutation_path: Path to Stage 2 permutation file.
        config: Configuration dictionary.
        output_dir: Directory to save checkpoint.
        use_simclr: If True, use SimCLR. If False, use supervised.
        device: Device to train on.

    Returns:
        Path to saved checkpoint.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get layer configs
    layer_configs = config['model']['layers']
    train_config = config.get('training', {})
    alpha = train_config.get('spatial_loss_weight_alpha', 0.25)

    # Create TDANN model
    print("Creating TDANN model...")
    model = create_tdann(
        layer_configs=layer_configs,
        use_simclr=use_simclr,
        spatial_loss_alpha=alpha,
        pretrained_backbone=False
    )

    # Step 1: Load permutations from Stage 2
    print(f"Loading permutations from {permutation_path}")
    permutations = torch.load(permutation_path)
    for layer_id, perm in permutations.items():
        model.set_layer_permutation(layer_id, perm)

    # Step 2: Reset all weights (do NOT use Stage 1 weights)
    print("Resetting all backbone weights...")
    model.reset_backbone_weights()

    model = model.to(device)

    # Training configuration
    batch_size = train_config.get('batch_size', 128)
    epochs = train_config.get('epochs_stage3', 200)
    lr = train_config.get('learning_rate', 0.6)
    weight_decay = train_config.get('weight_decay', 1e-4)

    # Create dataloader
    if use_simclr:
        dataloader = create_simclr_dataloader(
            hf_dataset,
            batch_size=batch_size,
            num_workers=4
        )
        simclr_criterion = SimCLRLoss(temperature=0.5)
    else:
        dataloader = create_standard_dataloader(
            hf_dataset,
            batch_size=batch_size,
            num_workers=4,
            is_train=True
        )
        ce_criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay
    )

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    print(f"\nStarting Stage 3 training for {epochs} epochs...")
    print(f"Mode: {'SimCLR' if use_simclr else 'Supervised'}")
    print(f"Spatial Loss Weight (alpha): {alpha}")
    print(f"Device: {device}")

    checkpoint_path = output_dir / 'tdann_final.pt'

    for epoch in range(1, epochs + 1):
        if use_simclr:
            metrics = train_epoch_simclr(
                model, dataloader, optimizer, simclr_criterion, device, epoch
            )
        else:
            metrics = train_epoch_supervised(
                model, dataloader, optimizer, ce_criterion, device, epoch
            )

        scheduler.step()

        print(f"Epoch {epoch}/{epochs} | "
              f"Total: {metrics['total_loss']:.4f} | "
              f"Task: {metrics['task_loss']:.4f} | "
              f"Spatial: {metrics['spatial_loss']:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint periodically
        if epoch % 10 == 0 or epoch == epochs:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'metrics': metrics,
                'use_simclr': use_simclr,
                'alpha': alpha
            }, checkpoint_path)

    print(f"\nStage 3 complete. Final model saved to {checkpoint_path}")
    return str(checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description='Stage 3: Final Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to HuggingFace dataset cache')
    parser.add_argument('--permutation-path', type=str, required=True,
                        help='Path to Stage 2 permutation file')
    parser.add_argument('--output-dir', type=str, default='outputs/stage3',
                        help='Output directory for final model')
    parser.add_argument('--use-simclr', action='store_true', default=True,
                        help='Use SimCLR (default: True)')
    parser.add_argument('--supervised', action='store_true',
                        help='Use supervised training instead of SimCLR')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load dataset
    from datasets import load_dataset
    print(f"Loading dataset from {args.dataset_path}...")
    hf_dataset = load_dataset('timm/mini-imagenet', cache_dir=args.dataset_path)

    # Run Stage 3
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_simclr = not args.supervised

    run_stage3(
        hf_dataset=hf_dataset['train'],
        permutation_path=args.permutation_path,
        config=config,
        output_dir=args.output_dir,
        use_simclr=use_simclr,
        device=device
    )


if __name__ == '__main__':
    main()
