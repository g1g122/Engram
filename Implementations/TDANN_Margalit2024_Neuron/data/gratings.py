"""
Sine Gratings Generator for TDANN.

Responsibility:
    - Generate sine grating images for Stage 2 (Shuffling) and V1 Analysis.
    - Parameters: Orientation (0-180°), Spatial Frequency (cpd), Phase, Chromaticity.
    - Used to compute responses from Stage 1 model for position shuffling.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional


def generate_sine_grating(
    size: int = 224,
    orientation: float = 0.0,
    spatial_freq: float = 1.0,
    phase: float = 0.0,
    chromaticity: str = 'bw',
    visual_field_dva: float = 8.0
) -> np.ndarray:
    """
    Generate a single sine grating image.

    Args:
        size: Image size in pixels (assumes square image).
        orientation: Grating orientation in degrees [0, 180).
        spatial_freq: Spatial frequency in cycles per degree (cpd).
        phase: Spatial phase in radians [0, 2π).
        chromaticity: 'bw' for black/white, 'rc' for red/cyan.
        visual_field_dva: Visual field extent in degrees of visual angle.

    Returns:
        RGB image as numpy array of shape (size, size, 3), values in [0, 1].
    """
    # Convert orientation to radians
    theta = np.radians(orientation)

    # Create coordinate grids in degrees of visual angle
    extent = visual_field_dva / 2.0  # ±4 dva for 8 dva field
    x = np.linspace(-extent, extent, size)
    y = np.linspace(-extent, extent, size)
    X, Y = np.meshgrid(x, y)

    # Rotate coordinates
    X_rot = X * np.cos(theta) + Y * np.sin(theta)

    # Generate sine wave
    # grating = sin(2π * freq * x + phase)
    grating = np.sin(2 * np.pi * spatial_freq * X_rot + phase)

    # Normalize to [0, 1]
    grating = (grating + 1) / 2

    # Create RGB image based on chromaticity
    if chromaticity == 'bw':
        # Black and white grating
        img = np.stack([grating, grating, grating], axis=2)
    elif chromaticity == 'rc':
        # Red-cyan grating
        # Red channel: grating, Green/Blue: inverted
        red = grating
        cyan = 1 - grating
        img = np.stack([red, cyan, cyan], axis=2)
    else:
        raise ValueError(f"Unknown chromaticity: {chromaticity}. Use 'bw' or 'rc'.")

    return img.astype(np.float32)


class SineGratingDataset(Dataset):
    """
    Dataset of sine grating images with systematic parameter variation.

    Generates all combinations of orientations, spatial frequencies, phases,
    and chromaticities as specified in the paper.
    """

    def __init__(
        self,
        img_size: int = 224,
        orientations: Optional[List[float]] = None,
        spatial_freqs: Optional[List[float]] = None,
        phases: Optional[List[float]] = None,
        chromaticities: Optional[List[str]] = None,
        visual_field_dva: float = 8.0
    ):
        """
        Args:
            img_size: Image size in pixels.
            orientations: List of orientations in degrees. 
                          Default: 8 evenly spaced from 0 to 180.
            spatial_freqs: List of spatial frequencies in cpd.
                           Default: 8 values from 0.5 to 12 cpd.
            phases: List of phases in radians.
                    Default: 5 evenly spaced from 0 to 2π.
            chromaticities: List of chromaticity types.
                            Default: ['bw', 'rc'].
            visual_field_dva: Visual field extent in degrees.
        """
        super().__init__()
        self.img_size = img_size
        self.visual_field_dva = visual_field_dva

        # Default parameters from paper (lines 1641-1643)
        if orientations is None:
            # 8 orientations evenly spaced between 0 and 180 degrees
            orientations = np.linspace(0, 180, 8, endpoint=False).tolist()
        if spatial_freqs is None:
            # 8 spatial frequencies between 0.5 and 12 cpd
            # Use log spacing for more even perceptual distribution
            spatial_freqs = np.logspace(np.log10(0.5), np.log10(12), 8).tolist()
        if phases is None:
            # 5 spatial phases evenly spaced from 0 to 2π
            phases = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist()
        if chromaticities is None:
            # Two chromaticities: black/white and red/cyan
            chromaticities = ['bw', 'rc']

        self.orientations = orientations
        self.spatial_freqs = spatial_freqs
        self.phases = phases
        self.chromaticities = chromaticities

        # Generate all combinations
        self.params = []
        for ori in orientations:
            for sf in spatial_freqs:
                for ph in phases:
                    for chrom in chromaticities:
                        self.params.append({
                            'orientation': ori,
                            'spatial_freq': sf,
                            'phase': ph,
                            'chromaticity': chrom
                        })

    def __len__(self) -> int:
        return len(self.params)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Get a single grating image and its parameters.

        Returns:
            Tuple of:
                - image: Tensor of shape (3, H, W), normalized to [0, 1]
                - params: Dict with grating parameters
        """
        params = self.params[idx]

        # Generate grating
        img = generate_sine_grating(
            size=self.img_size,
            orientation=params['orientation'],
            spatial_freq=params['spatial_freq'],
            phase=params['phase'],
            chromaticity=params['chromaticity'],
            visual_field_dva=self.visual_field_dva
        )

        # Convert to tensor: (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)

        return img_tensor, params


def create_grating_dataset(
    img_size: int = 224,
    num_orientations: int = 8,
    num_spatial_freqs: int = 8,
    num_phases: int = 5,
    visual_field_dva: float = 8.0
) -> SineGratingDataset:
    """
    Factory function to create a grating dataset with default paper parameters.

    Args:
        img_size: Image size in pixels.
        num_orientations: Number of orientations to sample.
        num_spatial_freqs: Number of spatial frequencies to sample.
        num_phases: Number of phases to sample.
        visual_field_dva: Visual field extent in degrees.

    Returns:
        SineGratingDataset instance.
    """
    # Generate parameter ranges
    orientations = np.linspace(0, 180, num_orientations, endpoint=False).tolist()
    spatial_freqs = np.logspace(np.log10(0.5), np.log10(12), num_spatial_freqs).tolist()
    phases = np.linspace(0, 2 * np.pi, num_phases, endpoint=False).tolist()
    chromaticities = ['bw', 'rc']

    return SineGratingDataset(
        img_size=img_size,
        orientations=orientations,
        spatial_freqs=spatial_freqs,
        phases=phases,
        chromaticities=chromaticities,
        visual_field_dva=visual_field_dva
    )
