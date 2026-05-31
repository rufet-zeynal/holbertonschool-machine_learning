#!/usr/bin/env python3
"""Gaussian Process - adds prediction"""
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
        """RBF kernel — same as task 0"""
        # replace the sqdist block with this
        sum_x1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        sum_x2 = np.sum(X2 ** 2, axis=1)
        sqdist = sum_x1 + sum_x2 - 2 * np.dot(X1, X2.T)
        return self.sigma_f ** 2 * np.exp(-sqdist / (2 * self.l ** 2))

    def predict(self, X_s):
        """
        Posterior mean and variance at test points X_s.

        K_s  = cov between training X and test X_s  -> shape (t, s)
        K_ss = cov of test points with themselves    -> shape (s, s)

        mu    = K_s.T @ K^{-1} @ Y
        sigma = diag(K_ss - K_s.T @ K^{-1} @ K_s)

        Using linalg.solve(K, K_s) instead of inv(K) @ K_s
        avoids explicitly inverting K, which is slow and unstable.
        """
        K_s = self.kernel(self.X, X_s)       # (t, s)
        K_ss = self.kernel(X_s, X_s)         # (s, s)

        # solve K @ alpha = K_s  =>  alpha = K^{-1} @ K_s
        K_inv_Ks = np.linalg.solve(self.K, K_s)  # (t, s)

        # mean: shape (s,)
        mu = (K_inv_Ks.T @ self.Y).reshape(-1)

        # variance: take only the diagonal of the s×s matrix
        sigma = np.diag(K_ss - K_s.T @ K_inv_Ks)  # (s,)

        return mu, sigma
