#!/usr/bin/env python3
"""Identity block for ResNet"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in
    Deep Residual Learning for Image Recognition (2015)

    A_prev: output from the previous layer
    filters: tuple/list containing F11, F3, F12
    Returns: activated output of the identity block
    """
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    # Main path
    # First component: 1x1 conv
    X = K.layers.Conv2D(F11, (1, 1), padding='same',
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

    # Shortcut: add input directly (shapes match — that's why it's "identity")
    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)

    return X
