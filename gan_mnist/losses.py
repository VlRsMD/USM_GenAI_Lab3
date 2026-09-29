"""GAN loss functions based on binary cross-entropy."""

import tensorflow as tf

_bce = tf.keras.losses.BinaryCrossentropy()


def discriminator_loss(real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
    """D should output 1 for real images and 0 for generated ones.

    d_loss = BCE(1, D(x)) + BCE(0, D(G(z))). At equilibrium (D = 0.5 everywhere)
    this is 2 * ln 2 ~= 1.386.
    """
    real_loss = _bce(tf.ones_like(real_output), real_output)
    fake_loss = _bce(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output: tf.Tensor) -> tf.Tensor:
    """G wants its fakes classified as real: g_loss = BCE(1, D(G(z))) = -log D(G(z)).

    This is the non-saturating form of the minimax loss; it gives strong
    gradients early in training when D easily rejects the fakes. At
    equilibrium it is ln 2 ~= 0.693.
    """
    return _bce(tf.ones_like(fake_output), fake_output)
