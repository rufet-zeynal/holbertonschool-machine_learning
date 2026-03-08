#!/usr/bin/env python3
"""
Module for a single neuron with private attributes
"""
import numpy as np


class Neuron:
    """
    Neuron class
    """
    def __init__(self, nx):
        """
        Class constructor for the Neuron with private attributes
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
