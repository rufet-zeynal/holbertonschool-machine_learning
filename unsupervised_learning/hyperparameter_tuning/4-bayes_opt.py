#!/usr/bin/env python3
"""Bayesian Optimization - Expected Improvement acquisition"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Bayesian optimization over a noiseless 1D GP"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        min_b, max_b = bounds
        self.X_s = np.linspace(min_b, max_b, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Expected Improvement at every candidate in X_s.

        For minimization:  f_best = min(Y),  improvement = f_best - mu
        For maximization:  f_best = max(Y),  improvement = mu - f_best

        EI(x) = improvement · Phi(Z) + sigma · phi(Z)
        where Z = improvement / sigma
        and Phi, phi are the normal CDF and PDF.

        Points where sigma == 0 get EI = 0 (we already know the answer there).
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            f_best = np.min(self.gp.Y)
            improvement = f_best - mu - self.xsi
        else:
            f_best = np.max(self.gp.Y)
            improvement = mu - f_best - self.xsi

        # avoid division by zero — mask out flat spots
        Z = np.where(sigma > 0, improvement / sigma, 0.0)

        EI = np.where(
            sigma > 0,
            improvement * norm.cdf(Z) + sigma * norm.pdf(Z),
            0.0
        )

        X_next = self.X_s[np.argmax(EI)].reshape(-1)
        return X_next, EI
