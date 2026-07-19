#!/usr/bin/env python3
"""KL divergence cost for the t-SNE transformation."""
import numpy as np


def cost(P, Q):
    """
    Computes the KL divergence C = sum_ij P_ij * log(P_ij / Q_ij).
    """
    # Guard against log(0) by clamping both matrices from below at ~0.
    P_safe = np.maximum(P, 1e-12)
    Q_safe = np.maximum(Q, 1e-12)

    # KL divergence: sum over all pairs (including i=j, but those are 0).
    C = np.sum(P_safe * np.log(P_safe / Q_safe))

    return C
