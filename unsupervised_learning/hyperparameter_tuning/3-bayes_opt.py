#!/usr/bin/env python3
"""Bayesian Optimization shell"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Bayesian optimization over a noiseless 1D GP"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)

        # candidate points where EI will be evaluated — evenly spaced grid
        min_b, max_b = bounds
        self.X_s = np.linspace(min_b, max_b, ac_samples).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize
