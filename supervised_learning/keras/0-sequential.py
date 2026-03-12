#!/usr/bin/env python3
"""
Building a neural network by using Keras Sequential API
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Neural network with Keras library
    nx - number of neurons in each layer
    layers - list of nodes in each layer
    activations - list of activation functions in each layer
    lambtha - lambda parameter L2 regularization
    keep_prob - dropout rate
    """
    model = K.Sequential()

    for i in range(len(layers)):
        model.add(K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha),
            input_shape=(nx,) if i == 0 else ()
        ))

        if i < len(layers) - 1:
            model.add(K.layers.Dropout(rate=1-keep_prob))

    return model
