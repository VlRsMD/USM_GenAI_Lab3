"""Train on MNIST and save sample grids, loss curves and models.

Usage:
    python main.py                 # full run: 30 epochs
    python main.py --epochs 15     # shorter run if training is slow
    python main.py --show          # also open the figures in a window
"""

import argparse
import dataclasses
import logging
from pathlib import Path

from gan_mnist.config import GANConfig
from gan_mnist.data import load_mnist_dataset
from gan_mnist.models import build_discriminator, build_generator
from gan_mnist.trainer import GANTrainer
from gan_mnist.utils import save_history, set_seed, setup_logging
from gan_mnist.visualization import plot_image_grid, plot_losses

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    defaults = GANConfig()
    parser = argparse.ArgumentParser(description="Train a Dense GAN on MNIST (Laborator 3).")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--preview-every", type=int, default=defaults.preview_every)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--show", action="store_true", help="display figures as well as saving them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = dataclasses.replace(
        GANConfig(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        preview_every=args.preview_every,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    setup_logging()
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    set_seed(config.seed)

    # Step 0: data
    dataset = load_mnist_dataset(config)

    # Steps 1-2: models
    generator = build_generator(config)
    discriminator = build_discriminator(config)
    generator.summary()
    discriminator.summary()

    # Step 3: training, with a 4x4 preview from the same fixed noise every few epochs
    trainer = GANTrainer(generator, discriminator, config)
    fixed_noise = trainer.sample_noise(config.preview_samples)

    def save_preview(epoch: int, t: GANTrainer) -> None:
        if epoch % config.preview_every == 0 or epoch == 1:
            plot_image_grid(
                t.generator(fixed_noise, training=False).numpy(),
                image_size=config.image_size,
                title=f"Epoch {epoch}",
                save_path=config.samples_dir / f"epoch_{epoch:03d}.png",
                show=args.show,
            )

    history = trainer.fit(dataset, on_epoch_end=save_preview)

    # Step 4: 25 digits from fresh noise
    final_images = generator(trainer.sample_noise(config.final_samples), training=False).numpy()
    plot_image_grid(
        final_images,
        image_size=config.image_size,
        title="Generated digits (final)",
        save_path=config.output_dir / "final_grid.png",
        show=args.show,
    )
    plot_losses(history.d_loss, history.g_loss, save_path=config.output_dir / "losses.png", show=args.show)
    save_history(history, config.output_dir / "history.json")

    config.models_dir.mkdir(parents=True, exist_ok=True)
    generator.save(config.models_dir / "generator.keras")
    discriminator.save(config.models_dir / "discriminator.keras")

    logger.info("Done. Results saved in %s", config.output_dir.resolve())


if __name__ == "__main__":
    main()
