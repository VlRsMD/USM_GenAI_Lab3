"""Fast sanity checks for shapes, value ranges and one training step."""

import math

import numpy as np
import pytest
import tensorflow as tf

from gan_mnist.config import GANConfig
from gan_mnist.data import denormalize_images, normalize_images
from gan_mnist.losses import discriminator_loss, generator_loss
from gan_mnist.models import build_discriminator, build_generator
from gan_mnist.trainer import GANTrainer


@pytest.fixture
def config() -> GANConfig:
    return GANConfig(batch_size=8)


def test_normalization_round_trip():
    pixels = np.array([0, 127.5, 255], dtype="float32")
    normalized = normalize_images(pixels)
    np.testing.assert_allclose(normalized, [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(denormalize_images(normalized), [0.0, 0.5, 1.0])


def test_generator_output_shape_and_range(config):
    generator = build_generator(config)
    images = generator(tf.random.normal((4, config.latent_dim))).numpy()
    assert images.shape == (4, 784)
    assert images.min() >= -1.0 and images.max() <= 1.0


def test_discriminator_outputs_probabilities(config):
    discriminator = build_discriminator(config)
    probs = discriminator(tf.random.uniform((4, 784), -1.0, 1.0)).numpy()
    assert probs.shape == (4, 1)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_losses_at_equilibrium():
    half = tf.fill((16, 1), 0.5)
    assert float(discriminator_loss(half, half)) == pytest.approx(2 * math.log(2), rel=1e-4)
    assert float(generator_loss(half)) == pytest.approx(math.log(2), rel=1e-4)


def test_train_step_updates_both_networks(config):
    trainer = GANTrainer(build_generator(config), build_discriminator(config), config)
    g_before = [w.numpy().copy() for w in trainer.generator.trainable_variables]
    d_before = [w.numpy().copy() for w in trainer.discriminator.trainable_variables]

    d_loss, g_loss = trainer.train_step(tf.random.uniform((config.batch_size, 784), -1.0, 1.0))

    assert np.isfinite(float(d_loss)) and np.isfinite(float(g_loss))
    assert any(not np.allclose(a, b.numpy()) for a, b in zip(g_before, trainer.generator.trainable_variables))
    assert any(not np.allclose(a, b.numpy()) for a, b in zip(d_before, trainer.discriminator.trainable_variables))
