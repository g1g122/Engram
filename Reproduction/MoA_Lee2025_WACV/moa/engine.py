import torch

from moa.losses import total_classification_loss
from moa.utils import MetricLogger, accuracy, move_to_device


def _device_type(device):
    """Return device type string accepted by torch.amp.autocast."""
    if isinstance(device, torch.device):
        return device.type
    return torch.device(device).type


def unpack_batch(batch):
    """Extract images and labels from common batch formats.

    Supports:
        (images, labels)
        (images, labels, ...)
        {"image": images, "label": labels}
        {"images": images, "labels": labels}
        {"x": images, "y": labels}
    """
    if isinstance(batch, dict):
        image_keys = ("image", "images", "x", "input", "inputs")
        label_keys = ("label", "labels", "y", "target", "targets")

        images = None
        labels = None

        for key in image_keys:
            if key in batch:
                images = batch[key]
                break

        for key in label_keys:
            if key in batch:
                labels = batch[key]
                break

        if images is None or labels is None:
            raise KeyError(
                "Could not find image/label tensors in batch dict. "
                f"Available keys: {list(batch.keys())}"
            )

        return images, labels

    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError("Batch tuple/list must contain at least images and labels.")
        return batch[0], batch[1]

    raise TypeError(f"Unsupported batch type: {type(batch).__name__}")


def merge_domain_batches(domain_batches):
    """Merge one batch from each source domain into a single train batch."""
    images = []
    labels = []

    for batch in domain_batches.values():
        batch_images, batch_labels = unpack_batch(batch)
        images.append(batch_images)
        labels.append(batch_labels)

    if not images:
        raise ValueError("domain_batches must contain at least one source batch.")

    return torch.cat(images, dim=0), torch.cat(labels, dim=0)


def _train_forward_loss(model, images, labels, aux_weight):
    logits = model(images)
    loss, loss_dict = total_classification_loss(
        logits,
        labels,
        model=model,
        aux_weight=aux_weight,
    )
    return logits, loss, loss_dict


def train_one_step(
    model,
    domain_batches,
    optimizer,
    device,
    aux_weight=0.0,
    scaler=None,
    max_grad_norm=None,
):
    """Run one DomainBed-style optimization step.

    Args:
        model: Classification model.
        domain_batches: Dict mapping source domain name to one batch.
        optimizer: Optimizer.
        device: torch device.
        aux_weight: Weight for MoA auxiliary load-balance loss.
        scaler: Optional torch.amp GradScaler.
        max_grad_norm: Optional gradient clipping norm.

    Returns:
        Dict of metrics for this optimization step.
    """
    model.train()
    domain_batches = move_to_device(domain_batches, device)
    images, labels = merge_domain_batches(domain_batches)

    optimizer.zero_grad(set_to_none=True)

    if scaler is not None:
        with torch.amp.autocast(device_type=_device_type(device)):
            logits, loss, loss_dict = _train_forward_loss(
                model,
                images,
                labels,
                aux_weight=aux_weight,
            )

        scaler.scale(loss).backward()

        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm,
            )

        scaler.step(optimizer)
        scaler.update()
    else:
        logits, loss, loss_dict = _train_forward_loss(
            model,
            images,
            labels,
            aux_weight=aux_weight,
        )
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm,
            )

        optimizer.step()

    (acc1,) = accuracy(logits.detach(), labels, topk=(1,))

    return {
        "loss": float(loss_dict["loss"]),
        "cls_loss": float(loss_dict["cls_loss"]),
        "aux_loss": float(loss_dict["aux_loss"]),
        "acc1": float(acc1.detach().cpu()),
        "batch_size": labels.size(0),
        "lr": optimizer.param_groups[0]["lr"],
    }


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    device,
    aux_weight=0.0,
):
    """Evaluate model.

    Args:
        model: Classification model.
        data_loader: Evaluation DataLoader.
        device: torch device.
        aux_weight: Weight for MoA auxiliary loss in reported total loss.

    Returns:
        Dict of averaged metrics.
    """
    model.eval()
    logger = MetricLogger()
    num_samples = 0

    for batch in data_loader:
        batch = move_to_device(batch, device)
        images, labels = unpack_batch(batch)

        logits = model(images)
        loss, loss_dict = total_classification_loss(
            logits,
            labels,
            model=model,
            aux_weight=aux_weight,
        )

        (acc1,) = accuracy(logits, labels, topk=(1,))
        batch_size = labels.size(0)
        num_samples += batch_size

        logger.update_with_count(
            batch_size,
            loss=loss_dict["loss"],
            cls_loss=loss_dict["cls_loss"],
            aux_loss=loss_dict["aux_loss"],
            acc1=acc1,
        )

    metrics = logger.averages()
    metrics["num_samples"] = num_samples
    return metrics
