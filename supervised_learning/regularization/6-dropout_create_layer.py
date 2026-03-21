#!/usr/bin/env python3
"""Create a Layer with Dropout"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Creates a layer of a neural network using dropout
    prev:      tensor output of previous layer
    n:         number of nodes in new layer
    activation: activation function
    keep_prob: probability that a node will be kept
    training:  boolean, whether model is in training mode
    Returns:   output of the new layer
    """
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_avg')
    )

    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)

    return dropout(layer(prev), training=training)
