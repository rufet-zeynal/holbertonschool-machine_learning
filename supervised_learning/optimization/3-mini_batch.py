#!/usr/bin/env python3
"""
Mini-batch
"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Mini Batch Creation
    X: numpy.ndarray of shape (m, nx)
    Y: numpy.ndarray of shape (m, ny)
    batch_size: number of data points in a batch
    """
    mini_batches = []
    m = X.shape[0]
    X, Y = shuffle_data(X, Y)

    for i in range(0, m, batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]
        mini_batches.append([X_batch, Y_batch])

    return mini_batches
