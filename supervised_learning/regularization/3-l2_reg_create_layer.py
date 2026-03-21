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
    regularizer = tf.keras.regularizers.L2(lambtha)

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=regularizer
    )

    return layer(prev)
