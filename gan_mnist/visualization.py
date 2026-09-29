"""Plotting helpers: image grids and loss curves."""

import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from gan_mnist.data import denormalize_images


def plot_image_grid(
    images: np.ndarray,
    image_size: int = 28,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Draw generated images (flat vectors in [-1, 1]) in a square grid."""
    images = denormalize_images(np.asarray(images)).reshape(-1, image_size, image_size)
    n = len(images)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig = plt.figure(figsize=(cols * 1.2, rows * 1.2))
    for i, image in enumerate(images):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    _finish(fig, save_path, show)


def plot_losses(
    d_losses: list[float],
    g_losses: list[float],
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Plot per-epoch discriminator and generator losses with the equilibrium levels."""
    epochs = range(1, len(d_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, d_losses, marker="o", label="Discriminator")
    ax.plot(epochs, g_losses, marker="o", label="Generator")
    ax.axhline(2 * math.log(2), color="C0", linestyle="--", linewidth=1, label="D equilibrium (2 ln 2)")
    ax.axhline(math.log(2), color="C1", linestyle="--", linewidth=1, label="G equilibrium (ln 2)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean BCE loss")
    ax.set_title("GAN training losses")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _finish(fig, save_path, show)


def _finish(fig: plt.Figure, save_path: Optional[Path], show: bool) -> None:
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
