#!/usr/bin/env python3
"""LeNet-5 architecture using Keras"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds modified LeNet-5 using Keras

    X - K.Input of shape (m, 28, 28, 1)

    Returns: compiled K.Model
    """
    # he_normal initializer with seed=0
    init = K.initializers.HeNormal(seed=0)

    # Conv layer 1 — 6 filters, 5×5, same padding
    x = K.layers.Conv2D(
        filters=6,
        kernel_size=5,
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(X)

    # Max pool 1 — 2×2, stride 2
    x = K.layers.MaxPooling2D(
        pool_size=2,
        strides=2
    )(x)

    # Conv layer 2 — 16 filters, 5×5, valid padding
    x = K.layers.Conv2D(
        filters=16,
        kernel_size=5,
        padding='valid',
        activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(x)

    # Max pool 2 — 2×2, stride 2
    x = K.layers.MaxPooling2D(
        pool_size=2,
        strides=2
    )(x)

    # Flatten — convert 2D to 1D
    x = K.layers.Flatten()(x)

    # Fully connected 1 — 120 nodes
    x = K.layers.Dense(
        120,
        activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(x)

    # Fully connected 2 — 84 nodes
    x = K.layers.Dense(
        84,
        activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(x)

    # Output — 10 classes (digits 0-9)
    output = K.layers.Dense(
        10,
        activation='softmax',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(x)

    # build model
    model = K.Model(inputs=X, outputs=output)

    # compile with Adam and accuracy
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
