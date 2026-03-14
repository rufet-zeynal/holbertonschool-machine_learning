#!/usr/bin/env python3
"""
Shuffle Data
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffle Data in two matrices same way
    """
    permutation = np.random.permutation(len(X))
    return X[permutation], Y[permutation]
