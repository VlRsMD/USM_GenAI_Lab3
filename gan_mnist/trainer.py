"""Adversarial training loop"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import tensorflow as tf

from gan_mnist.config import GANConfig
from gan_mnist.losses import discriminator_loss, generator_loss

logger = logging.getLogger(__name__)

EpochCallback = Callable[[int, "GANTrainer"], None]


@dataclass
class TrainingHistory:
    """Mean losses per epoch."""

    d_loss: list[float] = field(default_factory=list)
    g_loss: list[float] = field(default_factory=list)


class GANTrainer:
    """Trains a generator and discriminator against each other.

    Each step first updates D on a real batch plus a fake batch, then
    updates G through D with fresh noise, as the lab describes.
    """

    def __init__(
        self,
        generator: tf.keras.Model,
        discriminator: tf.keras.Model,
        config: GANConfig,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.g_optimizer = tf.keras.optimizers.Adam(config.learning_rate, beta_1=config.beta_1)
        self.d_optimizer = tf.keras.optimizers.Adam(config.learning_rate, beta_1=config.beta_1)
        self.history = TrainingHistory()

    def sample_noise(self, n: int) -> tf.Tensor:
        return tf.random.normal((n, self.config.latent_dim))

    @tf.function
    def train_step(self, real_images: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(real_images)[0]

        # 1. Discriminator: real -> 1, fake -> 0. G is not updated here.
        fake_images = self.generator(self.sample_noise(batch_size), training=True)
        with tf.GradientTape() as tape:
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            d_loss = discriminator_loss(real_output, fake_output)
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # 2. Generator: fresh noise, try to make D output 1. D is not updated here.
        with tf.GradientTape() as tape:
            fake_output = self.discriminator(
                self.generator(self.sample_noise(batch_size), training=True), training=True
            )
            g_loss = generator_loss(fake_output)
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return d_loss, g_loss

    def fit(
        self,
        dataset: tf.data.Dataset,
        epochs: Optional[int] = None,
        on_epoch_end: Optional[EpochCallback] = None,
    ) -> TrainingHistory:
        """Run the training loop over the full dataset for ``epochs`` epochs.

        ``on_epoch_end(epoch, trainer)`` is called after each epoch (1-based),
        e.g. to save preview grids.
        """
        epochs = epochs or self.config.epochs
        for epoch in range(1, epochs + 1):
            start = time.perf_counter()
            d_mean = tf.keras.metrics.Mean()
            g_mean = tf.keras.metrics.Mean()

            for real_images in dataset:
                d_loss, g_loss = self.train_step(real_images)
                d_mean.update_state(d_loss)
                g_mean.update_state(g_loss)

            self.history.d_loss.append(float(d_mean.result()))
            self.history.g_loss.append(float(g_mean.result()))
            logger.info(
                "Epoch %d/%d  d_loss=%.4f  g_loss=%.4f  (%.1fs)",
                epoch,
                epochs,
                self.history.d_loss[-1],
                self.history.g_loss[-1],
                time.perf_counter() - start,
            )

            if on_epoch_end is not None:
                on_epoch_end(epoch, self)

        return self.history
