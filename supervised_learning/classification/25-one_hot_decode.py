#!/usr/bin/env python3
"""
Module to decode a one-hot encoded matrix into a label vector
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels.

    one_hot: a one-hot encoded numpy.ndarray with shape (classes, m)

    Returns: a numpy.ndarray with shape (m, ) containing the numeric
    labels for each example, or None on failure
    """
    # Validate the input to ensure it is a 2D numpy array
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    try:
        # np.argmax finds the index of the maximum value.
        # axis=0 means we check the maximum value going down each column.
        decoded = np.argmax(one_hot, axis=0)
        return decoded
    except Exception:
        return None
