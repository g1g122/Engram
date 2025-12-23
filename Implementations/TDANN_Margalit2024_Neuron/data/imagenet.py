"""
ImageNet/Mini-ImageNet Data Loader for TDANN.

Responsibility:
    - Load ImageNet-style dataset for training (supports HuggingFace datasets).
    - Apply SimCLR augmentations (two views per image) for self-supervised training.
    - Apply standard augmentations for supervised training.
    - Resize images to 224×224 as required by ResNet.
"""

from typing import Callable, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
import platform


# Module-level collate function (required for Windows multiprocessing)
def _simclr_collate_fn(batch):
    """Collate function for SimCLR that handles two views."""
    views, labels = zip(*batch)
    view1s, view2s = zip(*views)
    return (torch.stack(view1s), torch.stack(view2s)), torch.tensor(labels)


class SimCLRAugmentation:
    """
    SimCLR-style augmentation that generates two augmented views of an image.

    Standard SimCLR augmentations include:
        - Random resized crop
        - Random horizontal flip
        - Color jitter
        - Random grayscale
        - Gaussian blur
    """

    def __init__(self, img_size: int = 224):
        """
        Args:
            img_size: Target image size (224 for ResNet).
        """
        # Color jitter parameters from SimCLR paper
        color_jitter = T.ColorJitter(
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.2
        )

        self.transform = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate two augmented views of the same image.

        Args:
            img: PIL Image.

        Returns:
            Tuple of two tensors (view1, view2), each shape (3, 224, 224).
        """
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2


class StandardAugmentation:
    """
    Standard augmentation for supervised training.
    """

    def __init__(self, img_size: int = 224, is_train: bool = True):
        """
        Args:
            img_size: Target image size.
            is_train: If True, apply training augmentations. If False, only resize/crop.
        """
        if is_train:
            self.transform = T.Compose([
                T.RandomResizedCrop(img_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)


class HuggingFaceImageDataset(Dataset):
    """
    Wrapper for HuggingFace datasets with image data.

    Supports datasets like 'timm/mini-imagenet' loaded via load_dataset().
    """

    def __init__(
        self,
        hf_dataset,
        transform: Optional[Callable] = None,
        image_key: str = 'image',
        label_key: str = 'label',
        simclr_mode: bool = False
    ):
        """
        Args:
            hf_dataset: HuggingFace dataset object.
            transform: Transform to apply to images.
            image_key: Key for image column in dataset.
            label_key: Key for label column in dataset.
            simclr_mode: If True, return two augmented views. If False, return single view.
        """
        self.dataset = hf_dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key
        self.simclr_mode = simclr_mode

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        Get a single sample.

        Returns:
            If simclr_mode:
                Tuple of ((view1, view2), label)
            Else:
                Tuple of (image, label)
        """
        item = self.dataset[idx]
        img = item[self.image_key]

        # Ensure PIL Image
        if not isinstance(img, Image.Image):
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            else:
                img = Image.open(img)

        # Convert grayscale to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        label = item.get(self.label_key, -1)

        if self.transform is not None:
            if self.simclr_mode:
                view1, view2 = self.transform(img)
                return (view1, view2), label
            else:
                img = self.transform(img)
                return img, label

        return img, label


def create_simclr_dataloader(
    hf_dataset,
    batch_size: int = 128,
    num_workers: int = 4,
    img_size: int = 224,
    image_key: str = 'image',
    label_key: str = 'label'
) -> DataLoader:
    """
    Create a DataLoader for SimCLR training.

    Args:
        hf_dataset: HuggingFace dataset (e.g., from load_dataset()).
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        img_size: Target image size.
        image_key: Key for image column.
        label_key: Key for label column.

    Returns:
        DataLoader that yields ((view1_batch, view2_batch), labels).
    """
    augmentation = SimCLRAugmentation(img_size=img_size)

    dataset = HuggingFaceImageDataset(
        hf_dataset=hf_dataset,
        transform=augmentation,
        image_key=image_key,
        label_key=label_key,
        simclr_mode=True
    )

    # Use 0 workers on Windows to avoid pickle issues
    if platform.system() == 'Windows':
        num_workers = 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_simclr_collate_fn
    )


def create_standard_dataloader(
    hf_dataset,
    batch_size: int = 128,
    num_workers: int = 4,
    img_size: int = 224,
    is_train: bool = True,
    image_key: str = 'image',
    label_key: str = 'label'
) -> DataLoader:
    """
    Create a DataLoader for standard supervised training/evaluation.

    Args:
        hf_dataset: HuggingFace dataset.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        img_size: Target image size.
        is_train: If True, apply training augmentations.
        image_key: Key for image column.
        label_key: Key for label column.

    Returns:
        DataLoader that yields (images, labels).
    """
    augmentation = StandardAugmentation(img_size=img_size, is_train=is_train)

    dataset = HuggingFaceImageDataset(
        hf_dataset=hf_dataset,
        transform=augmentation,
        image_key=image_key,
        label_key=label_key,
        simclr_mode=False
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )
