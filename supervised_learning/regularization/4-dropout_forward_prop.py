#!/usr/bin/env python3
"""Forward Propagation with Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducting forward propagation using Dropout
    X:         numpy.ndarray of shape (nx, m)
    weights:   dictionary of weights and biases
    L:         number of layers
    keep_prob: probability that a node will be kept
    """
    cache = {'A0': X}
    # Previous layer activation:
    A = X
    for layer in range(1, L + 1):
        # Weights:
        W = weights["W{}".format(layer)]
        # Bias:
        b = weights["b{}".format(layer)]
        # Linear transformation:
        Z = W @ A + b
        if layer < L:
            # Activation:
            A = np.tanh(Z)
            # Dropout mask:
            D = np.random.rand(*A.shape) < keep_prob
            D = np.where(D, 1, 0)
            cache['D{}'.format(layer)] = D
            # Apply dropout mask:
            A *= D / keep_prob
        else:
            # No dropout on final layer; softmax activation:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        cache['A{}'.format(layer)] = A

    return cache
