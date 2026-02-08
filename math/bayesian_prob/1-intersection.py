#!/usr/bin/env python3
"""
Bayesian Probability Intersection
"""
import numpy as np


def factorial(x):
    """
    factorial function
    """
    if x == 0:
        return 1

    fact = 1
    for i in range(1, x + 1):
        fact *= i
    return fact


def intersection(x, n, P, Pr):
    """
    Intersection of P and Pr
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.ndim != P.shape:
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    c = factorial(n) / (factorial(n - x) * factorial(x))
    return (c * (P ** x) * ((1 - P) ** (n - x))) * Pr
