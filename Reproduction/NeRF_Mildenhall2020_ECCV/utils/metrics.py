"""Evaluation metrics for NeRF rendering quality."""

import torch


def mse2psnr(mse):
    """Convert Mean Squared Error to Peak Signal-to-Noise Ratio.

    PSNR = -10 · log10(MSE), assuming pixel values in [0, 1].
    Higher PSNR indicates better reconstruction quality.

    Args:
        mse: MSE loss value (scalar tensor or float).

    Returns:
        PSNR in dB (scalar tensor).
    """
    return -10.0 * torch.log10(torch.tensor(mse, dtype=torch.float32))
