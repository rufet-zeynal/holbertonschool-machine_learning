#!/usr/bin/env python3
"""
L2 Regularization Cost
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculating the regularization cost
    cost - cost without regularization
    lambtha - regularization parameter
    weights - dictionary of weights and biases
    L - number of layers
    m - number of nodes
    """
    l2_sum = 0
    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        l2_sum += np.sum(np.square(W))

    l2_cost = (lambtha / (2 * m)) * l2_sum

    return cost + l2_cost
