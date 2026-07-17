import torch
import torch.nn.functional as F

from moa.models import collect_moa_aux_loss


def classification_loss(logits, targets, reduction="mean"):
    """Standard cross-entropy classification loss."""
    return F.cross_entropy(logits, targets, reduction=reduction)


def moa_aux_loss(model, device=None):
    """Collect MoA auxiliary loss from all injected wrappers.

    Args:
        model: Model containing MoA wrappers.
        device: Optional device used when no aux loss is available.

    Returns:
        Scalar tensor. Returns 0 if no MoA aux loss is available.
    """
    aux_loss = collect_moa_aux_loss(model)

    if aux_loss is not None:
        return aux_loss

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    return torch.zeros((), device=device)


def total_classification_loss(
    logits,
    targets,
    model=None,
    aux_weight=0.0,
    reduction="mean",
):
    """Compute classification loss plus optional MoA auxiliary loss.

    Args:
        logits: Class logits with shape [B, num_classes].
        targets: Class labels with shape [B].
        model: Optional model containing MoA wrappers.
        aux_weight: Weight for MoA auxiliary load-balance loss.
        reduction: Cross-entropy reduction.

    Returns:
        total_loss: Scalar tensor used for backprop.
        loss_dict: Detached logging values.
    """
    cls_loss = classification_loss(logits, targets, reduction=reduction)

    if model is not None and aux_weight > 0:
        aux = moa_aux_loss(model, device=logits.device)
    else:
        aux = torch.zeros((), device=logits.device)

    total_loss = cls_loss + aux_weight * aux

    loss_dict = {
        "loss": total_loss.detach(),
        "cls_loss": cls_loss.detach(),
        "aux_loss": aux.detach(),
    }

    return total_loss, loss_dict
