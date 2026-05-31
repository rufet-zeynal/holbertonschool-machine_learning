#!/usr/bin/env python3
"""Gaussian Process - adds update"""
import numpy as np


class GaussianProcess:
    """Noiseless 1D Gaussian Process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """RBF kernel"""
        sum_x1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        sum_x2 = np.sum(X2 ** 2, axis=1)
        sqdist = sum_x1 + sum_x2 - 2 * np.dot(X1, X2.T)
        return self.sigma_f ** 2 * np.exp(-sqdist / (2 * self.l ** 2))

    def predict(self, X_s):
        """Posterior mean and variance at X_s"""
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv_Ks = np.linalg.solve(self.K, K_s)
        mu = (K_inv_Ks.T @ self.Y).reshape(-1)
        sigma = np.diag(K_ss - K_s.T @ K_inv_Ks)
        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Add a new sampled point and recompute K.
        X_new shape (1,) -> reshape to (1, 1) to stack with self.X (t, 1)
        Y_new shape (1,) -> reshape to (1, 1) to stack with self.Y (t, 1)
        """
        self.X = np.vstack((self.X, X_new.reshape(1, 1)))
        self.Y = np.vstack((self.Y, Y_new.reshape(1, 1)))
        # recompute K for the now-larger dataset
        self.K = self.kernel(self.X, self.X)
