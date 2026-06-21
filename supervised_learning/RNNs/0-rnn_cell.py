#!/usr/bin/env python3
"""
RNN Cell implementation.
"""
import numpy as np


class RNNCell:
    """
    Represents a cell of a simple RNN.
    """

    def __init__(self, i, h, o):
        """
        Initialize weights and biases.
        """
        # Weights (random normal distribution)
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)

        # Biases (initialized to zero)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Forward pass for a single time step.
        """
        # Combine previous memory and current input: shape (m, h + i)
        combined_state = np.concatenate((h_prev, x_t), axis=1)

        # Compute next hidden state with tanh activation
        h_next = np.tanh(np.matmul(combined_state, self.Wh) + self.bh)

        # Compute unnormalized output logits
        logits = np.matmul(h_next, self.Wy) + self.by

        # Compute numerically stable softmax probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return h_next, y
