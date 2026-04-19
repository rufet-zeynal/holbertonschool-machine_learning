#!/usr/bin/env python3
"""Projection block for ResNet"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in
    Deep Residual Learning for Image Recognition (2015)

    A_prev: output from the previous layer
    filters: tuple/list containing F11, F3, F12
    s: stride of the first convolution in both main path and shortcut
    Returns: activated output of the projection block
    """
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    # Main path
    # First component : 1x1 conv with stride s (downsamples)
    X = K.layers.Conv2D(F11, (1, 1), strides=(s, s), padding='same',
                        kernel_initializer=init)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second component: 3x3 conv
    X = K.layers.Conv2D(F3, (3, 3), padding='same',
                        kernel_initializer=init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third component: 1x1 conv (no activation yet)
    X = K.layers.Conv2D(F12, (1, 1), padding='same',
                        kernel_initializer=init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Shortcut path: 1x1 conv with stride s to match dimensions
    shortcut = K.layers.Conv2D(F12, (1, 1), strides=(s, s), padding='same',
                               kernel_initializer=init)(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # Add shortcut (no activation on shortcut before adding)
    X = K.layers.Add()([X, shortcut])
    X = K.layers.Activation('relu')(X)

    return X
