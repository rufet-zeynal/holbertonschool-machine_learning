#!/usr/bin/env python3
"""
Definiteness
"""
import numpy as np


def definiteness(matrix):
    """
    Definiteness of a matrix
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if (len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1] or
        matrix.size == 0):
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    eigenvalues = np.linalg.eigvals(matrix)
    positive = np.all(eigenvalues > 0)
    pos_semi = np.all(eigenvalues >= 0)
    negative = np.all(eigenvalues < 0)
    neg_semi = np.all(eigenvalues <= 0)

    if positive:
        return "Positive definite"
    if pos_semi:
        return "Positive semi-definite"
    if negative:
        return "Negative definite"
    if neg_semi:
        return "Negative semi-definite"

    if any(eigenvalues > 0) and any(eigenvalues < 0):
        return "Indefinite"

    return None
