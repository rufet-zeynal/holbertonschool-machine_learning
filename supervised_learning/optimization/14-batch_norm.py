#!/usr/bin/env python3
"""Batch normalization layer using TensorFlow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer

    prev       - activated output of previous layer
    n          - number of nodes in this layer
    activation - activation function to apply

    Returns: activated output tensor
    """
    # Dense layer with VarianceScaling initializer
    # NO activation here — apply it after batch norm
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'
        )
    )(prev)

    # Batch normalization
    # gamma initialized to 1 (scale)
    # beta  initialized to 0 (offset)
    # epsilon = 1e-7 to avoid division by zero
    batch_normed = tf.keras.layers.BatchNormalization(
        gamma_initializer=tf.keras.initializers.Ones(),   # gamma starts at 1
        beta_initializer=tf.keras.initializers.Zeros(),   # beta  starts at 0
        epsilon=1e-7
    )(dense)

    # NOW apply activation — after normalization
    output = activation(batch_normed)

    return output
