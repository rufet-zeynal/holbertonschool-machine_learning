#!/usr/bin/env python3
"""
Module to strictly one-hot encode a numeric label vector
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Function converting a numeric label vector into a one-hot matrix
    Y: numpy.ndarray with shape (m,) containing numeric class labels
    classes: maximum number of classes found in Y
    """
    # Validate the inputs to return None on failure
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if type(classes) is not int or classes <= np.amax(Y):
        return None

    try:
        # np.eye creates an identity matrix of size (classes x classes).
        # By indexing it with [Y], we extract the specific rows we need.
        # .T transposes it so that the examples are represented as columns.
        one_hot = np.eye(classes)[Y].T
        return one_hot
    except Exception:
        return None
