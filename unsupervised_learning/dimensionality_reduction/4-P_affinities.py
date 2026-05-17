#!/usr/bin/env python3
"""Symmetric P affinities for the full dataset via per-point binary search."""
import numpy as np

P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Computes the symmetric P affinities (t-SNE paper eq. 4).
    """
    n = X.shape[0]
    D, P, betas, H_target = P_init(X, perplexity)

    for i in range(n):
        # Distances from point i to every OTHER point.
        Di = np.concatenate([D[i, :i], D[i, i + 1:]])   # shape (n-1,)

        # Binary search bounds (None = not yet set → double/half instead).
        beta_lo = None   # lower bound on beta
        beta_hi = None   # upper bound on beta
        beta_i = betas[i, 0]

        Hi, Pi = HP(Di, beta_i)

        for _ in range(50):                              # max 50 iterations
            if np.abs(Hi - H_target) <= tol:
                break

            if Hi > H_target:
                # Entropy too high → distribution too flat → need larger beta
                # (sharper Gaussian, fewer effective neighbors).
                beta_lo = beta_i
                beta_i = beta_hi / 2 if beta_hi is not None else beta_i * 2
            else:
                # Entropy too low → distribution too peaked → need smaller beta.
                beta_hi = beta_i
                beta_i = beta_lo / 2 if beta_lo is not None else beta_i / 2

            Hi, Pi = HP(Di, beta_i)

        betas[i] = beta_i

        # Insert the (n-1) conditional probs back into the full row
        # (leaving the diagonal = 0).
        P[i, :i] = Pi[:i]
        P[i, i + 1:] = Pi[i:]

    # Symmetrize and normalize so the whole matrix sums to 1.
    P = (P + P.T) / (2 * n)

    return P
