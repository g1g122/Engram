"""
Stage 1: Pretrain ResNet-18 on ImageNet.

Responsibility:
    - Train a standard ResNet-18 on ImageNet (Self-Supervised SimCLR or Supervised).
    - Objective: Learn rich visual features that are "Task-General".
    - Output: A checkpoint of the pretrained model.
    - CRITICAL: This model is used ONLY to generate feature responses for Stage 2.
      Its weights are NOT transferred to Stage 3 (they are re-initialized).
    - Option to skip training and use torchvision pretrained weights directly.
"""

import argparse
import os
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

from models.backbone import ResNet18Backbone
from objectives.simclr_loss import SimCLRLoss, SimCLRProjector
from data.imagenet import create_simclr_dataloader, create_standard_dataloader


def train_simclr_epoch(
    model: nn.Module,
    projector: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: SimCLRLoss,
    device: torch.device,
    epoch: int
) -> float:
    """Train one epoch with SimCLR objective."""
    model.train()
    projector.train()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for (view1, view2), _ in pbar:
        view1 = view1.to(device)
        view2 = view2.to(device)

        optimizer.zero_grad()

        # Forward pass
        features1 = model(view1)
        features2 = model(view2)

        # Project to contrastive space
        z1 = projector(features1['pooled'])
        z2 = projector(features2['pooled'])

        # Compute loss
        loss = criterion(z1, z2)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / max(num_batches, 1)


def train_supervised_epoch(
    model: nn.Module,
    classifier: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    epoch: int
) -> float:
    """Train one epoch with supervised classification objective."""
    model.train()
    classifier.train()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        features = model(images)
        logits = classifier(features['pooled'])

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / max(num_batches, 1)


def run_stage1(
    hf_dataset,
    config: dict,
    output_dir: str,
    use_simclr: bool = True,
    use_pretrained: bool = False,
    device: Optional[torch.device] = None
) -> str:
    """
    Run Stage 1 pretraining.

    Args:
        hf_dataset: HuggingFace dataset for training.
        config: Configuration dictionary.
        output_dir: Directory to save checkpoint.
        use_simclr: If True, use SimCLR. If False, use supervised.
        use_pretrained: If True, skip training and use torchvision weights.
        device: Device to train on.

    Returns:
        Path to saved checkpoint.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    model = ResNet18Backbone(pretrained=use_pretrained)
    model = model.to(device)

    checkpoint_path = output_dir / 'stage1_checkpoint.pt'

    # If using pretrained weights, just save and return
    if use_pretrained:
        print("Using torchvision pretrained weights, skipping training...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'use_simclr': use_simclr,
            'use_pretrained': True,
            'epoch': 0
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        return str(checkpoint_path)

    # Training configuration
    train_config = config.get('training', {})
    batch_size = train_config.get('batch_size', 128)
    epochs = train_config.get('epochs_stage1', 200)
    lr = train_config.get('learning_rate', 0.6)
    weight_decay = train_config.get('weight_decay', 1e-4)

    # Create dataloader
    if use_simclr:
        dataloader = create_simclr_dataloader(
            hf_dataset,
            batch_size=batch_size,
            num_workers=4
        )
        projector = SimCLRProjector(input_dim=512).to(device)
        criterion = SimCLRLoss(temperature=0.5)

        # Optimizer for both model and projector
        params = list(model.parameters()) + list(projector.parameters())
        optimizer = optim.SGD(
            params,
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        dataloader = create_standard_dataloader(
            hf_dataset,
            batch_size=batch_size,
            num_workers=4,
            is_train=True
        )
        classifier = nn.Linear(512, 1000).to(device)
        criterion = nn.CrossEntropyLoss()

        params = list(model.parameters()) + list(classifier.parameters())
        optimizer = optim.SGD(
            params,
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    print(f"Starting Stage 1 training for {epochs} epochs...")
    print(f"Mode: {'SimCLR' if use_simclr else 'Supervised'}")
    print(f"Device: {device}")

    for epoch in range(1, epochs + 1):
        if use_simclr:
            avg_loss = train_simclr_epoch(
                model, projector, dataloader, optimizer, criterion, device, epoch
            )
        else:
            avg_loss = train_supervised_epoch(
                model, classifier, dataloader, optimizer, criterion, device, epoch
            )

        scheduler.step()

        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint periodically
        if epoch % 10 == 0 or epoch == epochs:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'use_simclr': use_simclr,
                'use_pretrained': False
            }, checkpoint_path)

    print(f"Stage 1 complete. Checkpoint saved to {checkpoint_path}")
    return str(checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description='Stage 1: Pretrain ResNet-18')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to HuggingFace dataset cache')
    parser.add_argument('--output-dir', type=str, default='outputs/stage1',
                        help='Output directory for checkpoint')
    parser.add_argument('--use-simclr', action='store_true', default=True,
                        help='Use SimCLR (default: True)')
    parser.add_argument('--supervised', action='store_true',
                        help='Use supervised training instead of SimCLR')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='Skip training and use torchvision pretrained weights')
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

    # Run Stage 1
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_simclr = not args.supervised

    run_stage1(
        hf_dataset=hf_dataset['train'],
        config=config,
        output_dir=args.output_dir,
        use_simclr=use_simclr,
        use_pretrained=args.use_pretrained,
        device=device
    )


if __name__ == '__main__':
    main()
