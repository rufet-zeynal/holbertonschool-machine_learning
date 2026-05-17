#!/usr/bin/env python3
"""PCA that retains a specified fraction of variance"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a zero-mean dataset, retaining `var`
    fraction of variance.
    """
    # SVD of X directly (X is already zero-mean per the task spec).
    # np.linalg.svd returns U (n×n), s (min(n,d),), Vt (d×d).
    # The right singular vectors Vt are the principal
    # components (eigenvectors
    # of X^T X), ordered by descending singular value / variance.
    _, s, Vt = np.linalg.svd(X)

    # Cumulative fraction of total singular-value mass.
    cumvar = np.cumsum(s) / np.sum(s)

    # First index where cumulative fraction meets or exceeds the target.
    nd = np.where(cumvar >= var)[0][0] + 1

    # W is the first nd rows of Vt transposed → shape (d, nd).
    W = Vt[:nd].T

    return W
