#!/usr/bin/env python3
"""
Normalization Constants
"""
import numpy as np


def normalization_constants(X):
    """
    Calculation of normalization constants
    """
    mean = np.mean(X)
    std = np.std(X)
    return mean, std
