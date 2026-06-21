#!/usr/bin/env python3
"""
RNN sequence forward pass.
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Forward pass for a simple RNN over all time steps.
    """
    # Get dimensions from input shape
    t, m, i = X.shape

    # Get output size from cell weights
    _, o = rnn_cell.Wy.shape
    h = h_0.shape[1]

    # Initialize storage arrays
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    # Set initial hidden state
    H[0] = h_0

    # Loop through each time step
    for step in range(t):
        # Get current input and previous memory
        x_t = X[step]
        h_prev = H[step]

        # Run one step forward through the cell
        h_next, y_t = rnn_cell.forward(h_prev, x_t)

        # Save results to storage arrays
        H[step + 1] = h_next
        Y[step] = y_t

    return H, Y
