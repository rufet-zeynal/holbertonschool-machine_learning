#!/usr/bin/env python3
"""
Making predictions with a neural network
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Making predictions with a neural network
    network - keras model
    data - input data
    verbose - print out during prediction
    """
    return network.predict(data, verbose=verbose)
