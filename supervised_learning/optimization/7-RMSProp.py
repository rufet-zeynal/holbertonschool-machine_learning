#!/usr/bin/env python3
"""
Updateing a variable using the RMSProp optimization algorithm from scratch.
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    RMSProp keeps a running average of the squared gradients to
    normalise the gradient, allowing a larger effective step size
    without diverging on steep dimensions.

    Formula:
        s_new  = beta2 * s + (1 - beta2) * grad²
        var_new = var - alpha * grad / (√s_new + epsilon)

    Args:
        alpha   (float):          Learning rate.
        beta2   (float):          RMSProp decay weight (typically 0.9).
        epsilon (float):          Small constant to avoid ÷0 (e.g. 1e-8).
        var     (numpy.ndarray):  Variable to update.
        grad    (numpy.ndarray):  Gradient of the loss w.r.t. var.
        s  (numpy.ndarray):  Previous second moment (squared-grad average).

    Returns:
        tuple: (updated_var, new_s)
            updated_var (numpy.ndarray): The updated variable.
            new_s       (numpy.ndarray): The updated second moment.
    """
    # Update the exponential moving average of squared gradients
    new_s = beta2 * s + (1 - beta2) * (grad ** 2)

    # Update the variable
    updated_var = var - alpha * (grad / (np.sqrt(new_s) + epsilon))

    return updated_var, new_s
