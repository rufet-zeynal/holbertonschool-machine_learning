#!/usr/bin/env python3
"""
Module to perform forward propagation for a deep RNN
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.
    """
    t, m, i = X.shape
    l, _, h = h_0.shape

    # Init hidden states array with shape (t + 1, l, m, h)
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    Y = []

    # Loop through time steps
    for step in range(t):
        current_input = X[step]

        # Loop through RNN layers
        for layer in range(l):
            cell = rnn_cells[layer]
            h_prev = H[step, layer]

            # Forward pass through current cell
            h_next, y_pred = cell.forward(h_prev, current_input)

            # Store hidden state for next time step
            H[step + 1, layer] = h_next

            # Hidden state becomes input for next layer
            current_input = h_next

        # Collect final layer output
        Y.append(y_pred)

    Y = np.array(Y)

    return H, Y
