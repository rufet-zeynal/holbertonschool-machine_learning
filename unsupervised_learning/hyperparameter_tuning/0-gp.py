#!/usr/bin/env python3
"""Gaussian Process with RBF kernel"""
import numpy as np


class GaussianProcess:
    """Noiseless 1D Gaussian Process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        # compute K right away using the initial samples
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        RBF (squared exponential) kernel.
        Trick: ||x1 - x2||^2 = sum(x1^2) + sum(x2^2) - 2*x1·x2
        This gives us the full (m, n) matrix in one shot without loops.
        """
        # sqdist shape: (m, n)
        sqdist = (
            np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            + np.sum(X2 ** 2, axis=1)
            - 2 * X1 @ X2.T
        )
        return self.sigma_f ** 2 * np.exp(-sqdist / (2 * self.l ** 2))
