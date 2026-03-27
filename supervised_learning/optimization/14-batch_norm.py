#!/usr/bin/env python3
"""Batch normalization layer using TensorFlow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates one layer with batch normalization

    prev       - output of previous layer
    n          - number of neurons
    activation - activation function

    Returns: activated output tensor
    """
    # Dense layer — keep default bias (don't set use_bias=False)
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'
        )
    )(prev)

    # Batch normalization with gamma=1, beta=0
    normed = tf.keras.layers.BatchNormalization(
        gamma_initializer=tf.keras.initializers.Ones(),
        beta_initializer=tf.keras.initializers.Zeros(),
        epsilon=1e-7
    )(dense, training=True)

    # Apply activation last
    output = activation(normed)

    return output
