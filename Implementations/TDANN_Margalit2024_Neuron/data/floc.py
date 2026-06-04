"""
fLoc (Functional Localizer) Data Loader for TDANN.

Responsibility:
    - Load fLoc dataset for VTC (Ventral Temporal Cortex) analysis.
    - Categories: faces, bodies, places, characters, objects (5 categories).
    - Each category has 2 subcategories, each with 144 images.
    - Used in 'analysis/vtc_metrics.py' to identify category-selective patches.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image


# Category mapping from subcategories to main categories
# Based on paper (line 1730-1732): faces, bodies, characters, places, objects
CATEGORY_MAPPING = {
    # Faces: adult and child faces
    'adult': 'faces',
    'child': 'faces',
    # Bodies: headless bodies and limbs
    'body': 'bodies',
    'limb': 'bodies',
    # Characters: pseudowords and numbers
    'word': 'characters',
    'number': 'characters',
    # Places: houses and corridors
    'house': 'places',
    'corridor': 'places',
    # Objects: string instruments and cars
    'instrument': 'objects',
    'car': 'objects',
}

# Main categories for VTC analysis
CATEGORIES = ['faces', 'bodies', 'characters', 'places', 'objects']
CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}


class FLocDataset(Dataset):
    """
    fLoc functional localizer dataset.

    The dataset contains grayscale images from 5 categories,
    each with 2 subcategories of 144 images each.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[T.Compose] = None,
        include_scrambled: bool = False
    ):
        """
        Args:
            root_dir: Path to fLoc dataset directory.
            transform: Transforms to apply to images.
            include_scrambled: Whether to include scrambled images (control).
        """
        self.root_dir = Path(root_dir)
        self.include_scrambled = include_scrambled

        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        # Collect all image paths and labels
        self.samples: List[Tuple[Path, str, str]] = []  # (path, subcategory, category)
        self._load_samples()

    def _load_samples(self):
        """Scan directories and collect image samples."""
        for subcat_dir in self.root_dir.iterdir():
            if not subcat_dir.is_dir():
                continue

            subcat_name = subcat_dir.name

            # Skip scrambled unless requested
            if subcat_name == 'scrambled' and not self.include_scrambled:
                continue

            # Get main category
            if subcat_name == 'scrambled':
                category = 'scrambled'
            elif subcat_name in CATEGORY_MAPPING:
                category = CATEGORY_MAPPING[subcat_name]
            else:
                continue  # Skip unknown directories

            # Collect image files
            for img_path in subcat_dir.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((img_path, subcat_name, category))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        """
        Get a single sample.

        Returns:
            Tuple of (image, category_idx, category_name, subcategory_name)
        """
        img_path, subcat, category = self.samples[idx]

        # Load image
        img = Image.open(img_path)

        # Convert grayscale to RGB (fLoc images are grayscale)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)

        # Get category index
        cat_idx = CATEGORY_TO_IDX.get(category, -1)

        return img, cat_idx, category, subcat


def create_floc_dataloader(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    include_scrambled: bool = False
) -> DataLoader:
    """
    Create a DataLoader for fLoc dataset.

    Args:
        root_dir: Path to fLoc dataset directory.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        include_scrambled: Whether to include scrambled control images.

    Returns:
        DataLoader yielding (images, category_indices, category_names, subcategory_names).
    """
    dataset = FLocDataset(
        root_dir=root_dir,
        include_scrambled=include_scrambled
    )

    def collate_fn(batch):
        imgs, cat_idxs, cats, subcats = zip(*batch)
        return (
            torch.stack(imgs),
            torch.tensor(cat_idxs),
            list(cats),
            list(subcats)
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for analysis
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )


def get_category_indices() -> Dict[str, int]:
    """Return mapping from category name to index."""
    return CATEGORY_TO_IDX.copy()


def get_num_categories() -> int:
    """Return number of main categories."""
    return len(CATEGORIES)
