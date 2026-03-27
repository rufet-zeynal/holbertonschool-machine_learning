#!/usr/bin/env python3
"""Adam optimization from scratch using NumPy"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable using Adam optimization

    alpha   - learning rate
    beta1   - first moment weight (direction memory)
    beta2   - second moment weight (size memory)
    epsilon - avoid division by zero
    var     - variable to update
    grad    - gradient of var
    v       - previous first moment
    s       - previous second moment
    t       - current time step (for bias correction)

    Returns: updated var, new v, new s
    """
    # Step 1 — update first moment (momentum/direction)
    v = beta1 * v + (1 - beta1) * grad

    # Step 2 — update second moment (adaptive step size)
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # Step 3 — bias correction
    # early in training v and s are too small, this fixes that
    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)

    # Step 4 — update the variable
    var = var - alpha * (v_corrected / (np.sqrt(s_corrected) + epsilon))

    return var, v, s
