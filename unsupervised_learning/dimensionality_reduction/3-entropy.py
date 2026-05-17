#!/usr/bin/env python3
"""Shannon entropy and conditional P affinities for a single data point."""
import numpy as np


def HP(Di, beta):
    """
    Computes the Shannon entropy Hi and conditional P affinities Pi for
    point i, given its pairwise distances Di and Gaussian precision beta.
    """
    # Unnormalized Gaussian kernel values.
    # Subtracting max(exponent) before exp prevents underflow without changing
    # the normalized result (the constant cancels in numerator/denominator).
    exponents = -Di * beta
    exponents -= np.max(exponents)          # numerical stability
    num = np.exp(exponents)

    # Normalize to get conditional probabilities.
    denom = np.sum(num)
    Pi = num / denom

    # Shannon entropy H = -sum p * log2(p).
    # Clip to avoid log(0); near-zero Pi values contribute ~0 anyway.
    Pi_safe = np.maximum(Pi, 1e-300)
    Hi = -np.sum(Pi_safe * np.log2(Pi_safe))

    return Hi, Pi
