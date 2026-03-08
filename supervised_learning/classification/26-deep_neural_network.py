#!/usr/bin/env python3
"""
Module defining a deep neural network performing binary classification
"""
import numpy as np
import pickle


def save(self, filename):
    """
    Saves the instance object to a file in pickle format

    filename: the file to which the object should be saved
    """
    if not filename.endswith('.pkl'):
        filename += '.pkl'

    with open(filename, 'wb') as file:
        pickle.dump(self, file)


@staticmethod
def load(filename):
    """
    Loads a pickled DeepNeuralNetwork object

    filename: the file from which the object should be loaded
    Returns: the loaded object, or None if filename doesn't exist
    """
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None
