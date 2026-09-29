"""MNIST loading and preprocessing (Laborator 3, step 0)."""

import numpy as np
import tensorflow as tf

from gan_mnist.config import GANConfig


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Scale uint8 pixels from [0, 255] to [-1, 1] to match the generator's tanh output."""
    return (images.astype("float32") - 127.5) / 127.5


def denormalize_images(images: np.ndarray) -> np.ndarray:
    """Map images from [-1, 1] back to [0, 1] for display."""
    return np.clip((images + 1.0) / 2.0, 0.0, 1.0)


def load_mnist_dataset(config: GANConfig) -> tf.data.Dataset:
    """Return the MNIST training images as a shuffled, batched dataset of flat vectors.

    Labels are discarded: a vanilla GAN learns the image distribution only.
    The last partial batch is dropped so every batch has exactly
    ``config.batch_size`` images.
    """
    (train_images, _), _ = tf.keras.datasets.mnist.load_data()
    images = normalize_images(train_images).reshape(-1, config.image_dim)

    return (
        tf.data.Dataset.from_tensor_slices(images)
        .shuffle(config.shuffle_buffer, seed=config.seed, reshuffle_each_iteration=True)
        .batch(config.batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
