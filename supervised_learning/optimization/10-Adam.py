#!/usr/bin/env python3
"""Adam optimization using TensorFlow"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Sets up Adam optimizer in TensorFlow

    alpha   - learning rate
    beta1   - first moment weight
    beta2   - second moment weight
    epsilon - avoid division by zero

    Returns: Adam optimizer object
    """
    return tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
