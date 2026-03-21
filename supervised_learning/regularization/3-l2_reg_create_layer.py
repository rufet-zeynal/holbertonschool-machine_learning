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
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    L2 = tf.contrib.layers.l2_regularizer(lambtha)
    # implements the operation: outputs = activation(inputs * kernel + bias)
    #                                                         weigth
    output = tf.layers.Dense(n, activation, kernel_initializer=init,
                             kernel_regularizer=L2)
    return output(prev)
