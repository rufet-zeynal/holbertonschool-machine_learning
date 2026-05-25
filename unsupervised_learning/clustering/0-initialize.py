#!/usr/bin/env python3
"""
Initialize cluster centroids for K-means clustering"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means using a multivariate
    uniform distribution.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None

    low = X.min(axis=0)   # shape (d,) - min per dimension
    high = X.max(axis=0)  # shape (d,) - max per dimension

    return np.random.uniform(low, high, size=(k, X.shape[1]))
