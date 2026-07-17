import random
from collections import defaultdict

import numpy as np
import torch


def set_seed(seed):
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(module):
    """Count all parameters in a module."""
    return sum(param.numel() for param in module.parameters())


def count_trainable_parameters(module):
    """Count trainable parameters in a module."""
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def get_trainable_parameter_names(module):
    """Return names of trainable parameters."""
    return [
        name
        for name, param in module.named_parameters()
        if param.requires_grad
    ]


def summarize_parameters(module):
    """Return a summary of total and trainable parameters."""
    total = count_parameters(module)
    trainable = count_trainable_parameters(module)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_ratio": trainable / total if total > 0 else 0.0,
    }


def move_to_device(batch, device):
    """Move tensors in a nested batch to device.

    Supports tensors, dicts, lists, and tuples.
    """
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)

    if isinstance(batch, dict):
        return {
            key: move_to_device(value, device)
            for key, value in batch.items()
        }

    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]

    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)

    return batch


@torch.no_grad()
def accuracy(logits, targets, topk=(1,)):
    """Compute top-k classification accuracy.

    Args:
        logits: Tensor with shape [B, num_classes].
        targets: Tensor with shape [B].
        topk: Tuple of k values.

    Returns:
        List of accuracies in percent.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, C], got {tuple(logits.shape)}")

    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.reshape(1, -1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append(correct_k.mul_(100.0 / batch_size))

    return results


class AverageMeter:
    """Track running average for scalar metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        if torch.is_tensor(value):
            value = value.detach().item()

        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


class MetricLogger:
    """Small metric logger backed by AverageMeter."""

    def __init__(self):
        self.meters = defaultdict(AverageMeter)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.meters[key].update(value)

    def update_with_count(self, count, **kwargs):
        for key, value in kwargs.items():
            self.meters[key].update(value, n=count)

    def averages(self):
        return {
            key: meter.avg
            for key, meter in self.meters.items()
        }

    def latest(self):
        return {
            key: meter.val
            for key, meter in self.meters.items()
        }

    def reset(self):
        self.meters.clear()
