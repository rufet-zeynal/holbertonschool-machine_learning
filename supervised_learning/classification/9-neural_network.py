#!/usr/bin/env python3
"""
Module defining a neural network with one hidden layer
performing binary classification.
"""
import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer
    performing binary classification.
    """
    def __init__(self, nx, nodes):
        """Constructor for NeuralNetwork"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Private Hidden Layer (Layer 1)
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Private Output Neuron (Layer 2)
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Weights 1 getter"""
        return self.__W1

    @property
    def b1(self):
        """Bias 1 getter"""
        return self.__b1

    @property
    def A1(self):
        """Activation 1 getter"""
        return self.__A1

    @property
    def W2(self):
        """Weights 2 getter"""
        return self.__W2

    @property
    def b2(self):
        """Bias 2 getter"""
        return self.__b2

    @property
    def A2(self):
        """Activation 2 getter"""
        return self.__A2
