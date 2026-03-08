#!/usr/bin/env python3
"""
Module defining a deep neural network performing binary classification
"""
import numpy as np
import pickle


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification
    """

    def __init__(self, nx, layers):
        # ... (all your code from task 23 goes here) ...
        pass

    # ... (all your properties, forward_prop, cost, etc.) ...

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        # ... (your train method from task 23) ...
        pass

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
