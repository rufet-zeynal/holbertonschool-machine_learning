#!/usr/bin/env python3
"""
GRU cell module.
"""
import numpy as np


class GRUCell:
    """
    Represents a gated recurrent unit.
    """

    def __init__(self, i, h, o):
        """
        Initialize the GRU cell weights and biases.
        """
        # Update gate weights and biases
        self.Wz = np.random.normal(size=(i + h, h))
        self.bz = np.zeros((1, h))

        # Reset gate weights and biases
        self.Wr = np.random.normal(size=(i + h, h))
        self.br = np.zeros((1, h))

        # Intermediate hidden state weights and biases
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))

        # Output weights and biases
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs a single forward pass step.
        """
        # Combine input and previous hidden state
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Compute update gate vector
        z_t = 1 / (1 + np.exp(-(np.dot(concat_input, self.Wz) + self.bz)))

        # Compute reset gate vector
        r_t = 1 / (1 + np.exp(-(np.dot(concat_input, self.Wr) + self.br)))

        # Combine reset memory with current input
        concat_reset = np.concatenate((r_t * h_prev, x_t), axis=1)

        # Compute candidate hidden state
        h_tilde = np.tanh(np.dot(concat_reset, self.Wh) + self.bh)

        # Calculate final hidden state
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # Calculate linear step for output
        y_linear = np.dot(h_next, self.Wy) + self.by

        # Apply softmax activation safely
        exp_y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, y
