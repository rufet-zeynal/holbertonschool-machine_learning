#!/usr/bin/env python3

"""Gradient Descent with L2 Regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updating weights and biases using gradient descent with L2 reg
    Y:       one-hot numpy.ndarray of shape (classes, m)
    weights: dictionary of weights and biases
    cache:   dictionary of outputs of each layer
    alpha:   learning rate
    lambtha: L2 regularization parameter
    L:       number of layers
    """
    m = Y.shape[1]
    # output layer gradient (softmax)
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        # weight gradient with L2 penalty
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        # bias gradient — no L2 penalty
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # backprop through tanh for previous layer
        if i > 1:
            dZ = np.matmul(W.T, dZ) * (1 - A_prev ** 2)

        # update weights and biases in place
        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] = b - alpha * db
