#!/usr/bin/env python3
"""Full t-SNE implementation with momentum and early exaggeration."""
import numpy as np

pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Performs t-SNE dimensionality reduction (van der Maaten & Hinton 2008).
    """
    n = X.shape[0]

    # --- Step 1: PCA pre-reduction ---
    X_pca = pca(X, idims)                             # (n, idims)

    # --- Step 2: P affinities in PCA space ---
    P = P_affinities(X_pca, perplexity=perplexity)   # (n, n)

    # --- Step 3: Initialize low-dim embedding ---
    # Small random initialization prevents early symmetry traps.
    Y = np.random.randn(n, ndims) * 1e-4              # (n, ndims)

    # Momentum arrays (iY = "previous update", used to carry momentum).
    iY = np.zeros_like(Y)   # Y_{t-1} - Y_{t-2}  (starts at 0)

    # --- Gradient descent ---
    for t in range(1, iterations + 1):

        # Momentum coefficient: 0.5 for first 20 steps, 0.8 thereafter.
        alpha = 0.5 if t <= 20 else 0.8

        # Early exaggeration: multiply P by 4 for the first 100 iterations.
        # This pushes tight clusters apart early, making the landscape easier
        # to escape from (see Section 3.4 of the paper).
        P_use = 4 * P if t <= 100 else P

        # Compute gradients and Q affinities.
        dY, Q = grads(Y, P_use)

        # Gradient descent with momentum (Algorithm 1, corrected sign):
        #   Y_new = Y - lr * dY + alpha * (Y - Y_prev)
        # The paper writes "+ alpha * iY" where iY = Y - Y_prev.
        # Note: the paper's Algorithm 1 has a sign error on the gradient term;
        # the correct update *subtracts* the gradient.
        iY = alpha * iY - lr * dY                     # momentum update
        Y = Y + iY

        # Re-center Y after every step (keeps the embedding stable).
        Y -= np.mean(Y, axis=0)

        # Print cost every 100 iterations.
        if t % 100 == 0:
            C = cost(P_use, Q)
            print(f"Cost at iteration {t}: {C}")

    return Y
