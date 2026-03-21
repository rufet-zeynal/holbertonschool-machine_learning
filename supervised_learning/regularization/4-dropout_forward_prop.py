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
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        # linear step
        Z = np.matmul(W, A_prev) + b

        if i == L:
            # last layer → softmax
            e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = e_Z / np.sum(e_Z, axis=0, keepdims=True)
        else:
            # hidden layers → tanh + dropout
            A = np.tanh(Z)
            mask = np.random.binomial(1, keep_prob, size=A.shape) / keep_prob
            A = A * mask
            cache['D' + str(i)] = mask

        cache['A' + str(i)] = A

    return cache
