#!/usr/bin/env python3
"""Gradient of the t-SNE cost (KL divergence) with respect to Y."""
import numpy as np

Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Computes the gradient dC/dY for the t-SNE cost C = KL(P || Q).
    """
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)

    # (P - Q) weighted by the Student-t kernel, shape (n, n).
    PQ = (P - Q) * num                      # element-wise

    # Gradient for each point i:
    # dY[i] = sum_j PQ[i,j] * (Y[i] - Y[j])
    #
    # Vectorized: PQ has shape (n, n), Y has shape (n, ndim).
    # For each i: dY[i] = sum_j PQ[i,j] * (Y[i] - Y[j])
    #           = PQ[i,:] @ Y[i] (broadcasted) - PQ[i,:] @ Y
    #
    # Compact form using einsum or broadcasting:
    #   dY = diag(PQ @ 1) * Y  -  PQ @ Y
    # where diag(PQ @ 1) is the row sums of PQ, used to scale Y[i].

    row_sums = np.sum(PQ, axis=1, keepdims=True)   # (n, 1)
    dY = row_sums * Y - np.dot(PQ, Y)              # (n, ndim)

    return dY, Q
