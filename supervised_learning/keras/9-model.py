#!/usr/bin/env python3
"""
Saves and loads Keras models
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saving an entire model to file
    network  - the Keras model to save
    filename - path to save the model to
    """
    network.save(filename)


def load_model(filename):
    """
    Loading an entire model from file
    """
    return K.models.load_model(filename)
