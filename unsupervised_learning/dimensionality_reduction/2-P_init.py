#!/usr/bin/env python3
"""Initialize variables needed for t-SNE P-affinity computation"""
import numpy as np


def P_init(X, perplexity):
    """
    Initializes D, P, betas, and H for t-SNE P-affinity computation
    """
    n = X.shape[0]

    # Broadcasting trick avoids an explicit double loop.
    sum_sq = np.sum(X ** 2, axis=1)           # shape (n,)
    # sum_sq[:, None] is (n,1), sum_sq[None, :] is (1,n)
    D = sum_sq[:, None] + sum_sq[None, :] - 2 * np.dot(X, X.T)
    # Numerical noise can make diagonal slightly non-zero; zero it explicitly.
    np.fill_diagonal(D, 0)

    P = np.zeros((n, n))
    betas = np.ones((n, 1))

    # Perplexity = 2^H  →  H = log2(perplexity)
    H = np.log2(perplexity)

    return D, P, betas, H
