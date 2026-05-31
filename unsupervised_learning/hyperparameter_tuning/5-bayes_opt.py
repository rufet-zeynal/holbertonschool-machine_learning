#!/usr/bin/env python3
"""Bayesian Optimization - full optimize loop"""
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
        """Expected Improvement acquisition"""
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            f_best = np.min(self.gp.Y)
            improvement = f_best - mu - self.xsi
        else:
            f_best = np.max(self.gp.Y)
            improvement = mu - f_best - self.xsi

        Z = np.where(sigma > 0, improvement / sigma, 0.0)
        EI = np.where(
            sigma > 0,
            improvement * norm.cdf(Z) + sigma * norm.pdf(Z),
            0.0
        )

        X_next = self.X_s[np.argmax(EI)].reshape(-1)
        return X_next, EI

    def optimize(self, iterations=100):
        """
        Run the Bayesian optimization loop.

        Each iteration:
          1. acquisition() -> find best candidate X_next
          2. check if X_next already sampled -> if yes, stop early
          3. evaluate black box: Y_next = f(X_next)
          4. update GP with the new (X_next, Y_next) pair

        After the loop, pick the best point from all sampled Y values.
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            # stop if this point was already sampled
            # np.isclose handles floating point comparison
            already_seen = np.any(np.isclose(self.gp.X, X_next))
            if already_seen:
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        # return the best point found across all samples
        if self.minimize:
            best_idx = np.argmin(self.gp.Y)
        else:
            best_idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[best_idx].reshape(-1)
        Y_opt = self.gp.Y[best_idx].reshape(-1)
        return X_opt, Y_opt
