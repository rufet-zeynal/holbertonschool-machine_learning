#!/usr/bin/env python3
"""
Sets up gradient descent with momentum using TensorFlow optimizer.
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Setting up gradient descent with momentum optimization in TensorFlow.
    alpha  (float): The learning rate.
    beta1  (float): The momentum weight (exponential decay rate for
                    the first moment estimate).
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
