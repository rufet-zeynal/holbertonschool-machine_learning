#!/usr/bin/env python3
"""
Task 8: RMSProp Upgraded
Sets up the RMSProp optimization algorithm using TensorFlow.
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in TensorFlow.

    Args:
        alpha (float): Learning rate.
        beta2 (float): RMSProp discounting factor
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        optimizer: A tf.keras.optimizers.RMSprop instance.
    """
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,   # rho is TF's name for beta2 / discount factor
        epsilon=epsilon
    )
    return optimizer
