#!/usr/bin/env python3
"""
Module for Neuron Cost
"""
import numpy as np


class Neuron:
    """
    Neuron class
    """
    def __init__(self, nx):
        """
        Class constructor for the Neuron
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    # Getter -> weights
    @property
    def W(self):
        """
        Getter for __W
        """
        return self.__W

    # Getter -> bias
    @property
    def b(self):
        """
        Getter for __b
        """
        return self.__b

    # Getter -> activation
    @property
    def A(self):
        """
        Getter for __A
        """
        return self.__A

    def forward_prop(self, X):
        """
        Forward propagation
        """
        # Z = W * X + b
        # Activation -> sigmoid(Z) = 1/(1+e^-Z)
        # W -> (1,nx), X -> (1, m) so WX = (1, m)
        Z = np.matmul(self.__W, X) + self.__b
        # Sigmoid function
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        Y: labels for the input data (1, m)
        A: activated output (1, m)
        """
        m = Y.shape[1]

        # We use 1.0000001 - A as requested to prevent
        # log(0) which is undefined
        # This is a common trick called "epsilon" to keep the math stable.
        cost = - (1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )

        return cost
