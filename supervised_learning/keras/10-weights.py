#!/usr/bin/env python3
"""
Saving and loads model weights
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saving a model's weights only
    network     - the Keras model whose weights to save
    filename    - path to save the weights to
    save_format - format to save weights in
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Loading weights into an existing model
    """
    network.load_weights(filename)
