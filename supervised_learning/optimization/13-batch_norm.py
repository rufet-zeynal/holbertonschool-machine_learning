#!/usr/bin/env python3
"""Batch normalization from scratch using NumPy"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output using batch normalization

    Z       - numpy array shape (m, n) to normalize
    gamma   - numpy array shape (1, n) scale parameters
    beta    - numpy array shape (1, n) offset parameters
    epsilon - small number to avoid division by zero

    Returns: normalized Z matrix
    """
    # Step 1 — mean of each feature (column)
    mean = np.mean(Z, axis=0, keepdims=True)

    # Step 2 — variance of each feature
    variance = np.var(Z, axis=0, keepdims=True)

    # Step 3 — normalize: zero mean, unit variance
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)

    # Step 4 — scale and shift with learned parameters
    Z_out = gamma * Z_norm + beta

    return Z_out
