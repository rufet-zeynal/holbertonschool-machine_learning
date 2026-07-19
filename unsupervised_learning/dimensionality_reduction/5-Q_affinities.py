#!/usr/bin/env python3
"""Q affinities in the low-dimensional t-SNE space (Student-t kernel)."""
import numpy as np


def Q_affinities(Y):
    """
    Computes Q affinities using the Student-t distribution (t-SNE paper eq. 4).
    """
    n = Y.shape[0]

    # Squared pairwise distances.
    sum_sq = np.sum(Y ** 2, axis=1)
    D_sq = sum_sq[:, None] + sum_sq[None, :] - 2 * np.dot(Y, Y.T)
    np.fill_diagonal(D_sq, 0)

    # Student-t kernel: num_ij = 1 / (1 + ||y_i - y_j||^2)
    num = 1.0 / (1.0 + D_sq)

    # Zero out diagonal (no self-similarity).
    np.fill_diagonal(num, 0)

    # Normalize.
    Q = num / np.sum(num)

    return Q, num
