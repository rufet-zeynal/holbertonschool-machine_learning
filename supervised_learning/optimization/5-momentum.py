#!/usr/bin/env python3
"""
GD with momentum
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updating a variable using GD with momentum
    alpha: learning rate
    beta1: momentum weight
    var: numpy.ndarray, variable to be updated
    grad: numpy.ndarray, gradient of var
    v: previous first moment of var
    """
    v_update = beta1 * v + (1 - beta1) * grad
    var_update = var - alpha * v_update

    return var_update, v_update
