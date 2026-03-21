#!/usr/bin/env python3
"""Creating a Layer with L2 Regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a neural network layer with L2 regularization
    prev:       tensor output of previous layer
    n:          number of nodes in new layer
    activation: activation function
    lambtha:    L2 regularization parameter
    """
    L2 = tf.keras.regularizers.L2(lambtha)
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))
    dense = tf.keras.layers.Dense(
        n, activation, kernel_regularizer=L2, kernel_initializer=init)
    return dense(prev)
