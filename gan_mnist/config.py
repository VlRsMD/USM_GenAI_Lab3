"""Hyperparameters and paths for the MNIST GAN (Laborator 3)."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GANConfig:
    """All settings for one training run.

    Defaults: latent_dim=100, batch 128,
    Adam(2e-4, beta_1=0.5) for both networks, 30 epochs, a 4x4 preview
    grid every 5 epochs and a final 5x5 grid.
    """

    # Data
    image_size: int = 28
    batch_size: int = 128
    shuffle_buffer: int = 60_000

    # Model
    latent_dim: int = 100
    leaky_relu_slope: float = 0.2

    # Optimisation
    learning_rate: float = 2e-4
    beta_1: float = 0.5
    epochs: int = 30

    # Monitoring
    preview_every: int = 5
    preview_samples: int = 16  # 4x4 grid from a fixed noise vector
    final_samples: int = 25  # 5x5 grid from fresh noise

    # Reproducibility and output
    seed: int = 42
    output_dir: Path = Path("outputs")

    @property
    def image_dim(self) -> int:
        """Length of a flattened image (28 * 28 = 784)."""
        return self.image_size * self.image_size

    @property
    def samples_dir(self) -> Path:
        return self.output_dir / "samples"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"
