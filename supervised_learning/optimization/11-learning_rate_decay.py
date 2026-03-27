#!/usr/bin/env python3
"""Learning rate decay using NumPy"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates learning rate using inverse time decay

    alpha       - original learning rate
    decay_rate  - how fast alpha decays
    global_step - how many gradient descent steps elapsed
    decay_step  - steps before each decay

    Returns: updated alpha
    """
    # floor division = stepwise (not smooth)
    alpha = alpha / (1 + decay_rate * np.floor(global_step / decay_step))

    return alpha
