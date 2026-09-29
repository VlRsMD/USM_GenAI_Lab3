"""Generator and discriminator networks.

Both are fully connected (Dense) networks.
"""

import tensorflow as tf
from tensorflow.keras import layers

from gan_mnist.config import GANConfig


def build_generator(config: GANConfig) -> tf.keras.Model:
    """z (100) -> Dense(256) -> Dense(512) -> Dense(784, tanh).

    The tanh output lies in [-1, 1], the same range as the normalized
    training images.
    """
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(config.latent_dim,)),
            layers.Dense(256),
            layers.LeakyReLU(negative_slope=config.leaky_relu_slope),
            layers.Dense(512),
            layers.LeakyReLU(negative_slope=config.leaky_relu_slope),
            layers.Dense(config.image_dim, activation="tanh"),
        ],
        name="generator",
    )


def build_discriminator(config: GANConfig) -> tf.keras.Model:
    """image (784) -> Dense(512) -> Dense(256) -> Dense(1, sigmoid).

    The output is the probability that the input image is real.
    """
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(config.image_dim,)),
            layers.Dense(512),
            layers.LeakyReLU(negative_slope=config.leaky_relu_slope),
            layers.Dense(256),
            layers.LeakyReLU(negative_slope=config.leaky_relu_slope),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )
