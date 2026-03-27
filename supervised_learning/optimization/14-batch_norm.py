#!/usr/bin/env python3
"""Batch normalization layer using TensorFlow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates one layer with batch normalization

    prev       - output of previous layer (tensor)
    n          - number of neurons in this layer
    activation - activation function (e.g. tf.nn.relu)

    Returns: activated output tensor
    """
    # Step 1 — Dense layer, no activation
    dense = tf.keras.layers.Dense(
        units=n,
        use_bias=False,    # BatchNorm has its own bias (beta)
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'
        )
    )(prev)

    # Step 2 — Batch normalization
    normed = tf.keras.layers.BatchNormalization(
        gamma_initializer=tf.keras.initializers.Ones(),
        beta_initializer=tf.keras.initializers.Zeros(),
        epsilon=1e-7
    )(dense)

    # Step 3 — apply activation last
    output = activation(normed)

    return output
