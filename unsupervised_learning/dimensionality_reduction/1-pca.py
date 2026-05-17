#!/usr/bin/env python3
"""PCA that reduces to a fixed number of dimensions, handling non-zero mean."""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset, projecting it to ndim dimensions.
    """
    # 1. Center the data (subtract column means).
    X_centered = X - np.mean(X, axis=0)

    # 2. SVD. Vt has shape (d, d); its rows are principal components.
    _, _, Vt = np.linalg.svd(X_centered)

    # 3. Keep the top ndim principal component directions.
    #    W has shape (d, ndim).
    W = Vt[:ndim].T

    # 4. Project: T = X_centered @ W  →  shape (n, ndim).
    T = np.matmul(X_centered, W)

    return T
