#!/usr/bin/env python3
"""
Normalize
"""
import numpy as np


def normalize(X, m, s):
    """
    Normalizing a matrix
    X is the numpy.ndarray of shape (d, nx)
    m is a mean of all features
    s is a standard deviation of all features of x
    """
    X_norm = (X - m) / s
    return X_norm
