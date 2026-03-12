#!/usr/bin/env python3
"""
Building a neural network using Keras Functional API
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Building a neural network without Sequential class
    nx          - number of input features
    layers      - list of nodes per layer
    activations - list of activation functions per layer
    lambtha     - L2 regularization parameter
    keep_prob   - probability that a node will be kept
    """
    inputs = K.layers.Input(shape=(nx,))

    x = K.layers.Dense(
        units=layers[0],
        activation=activations[0],
        kernel_regularizer=K.regularizers.L2(lambtha))(inputs)

    if len(layers) > 1:
        x = K.layers.Dropout(1 - keep_prob)(x)

    for i in range(1, len(layers)):
        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.L2(lambtha)
        )(x)

        if i < len(layers)-1:
            x = K.layers.Dropout(1-keep_prob)(x)

    model = K.models.Model(inputs=inputs, outputs=x)
    return model
