#!/usr/bin/env python3
"""
Saving and loads model configuration
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    saving a model's configuration to JSON file
    network  - the Keras model whose config to save
    filename - path to save the JSON config to
    """

    json_config = network.to_json()

    with open(filename, 'w') as f:
        f.write(json_config)


def load_config(filename):
    """
    loading a model from a JSON configuration file
    """

    with open(filename, 'r') as f:
        json_config = f.read()

    return K.models.model_from_json(json_config)
