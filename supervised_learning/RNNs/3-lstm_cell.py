#!/usr/bin/env python3
"""
Module containing the LSTMCell class
"""
import numpy as np


class LSTMCell:
    """
    Represents an LSTM unit
    """
    def __init__(self, i, h, o):
        """
        Class constructor for LSTMCell
        """
        # Init gate and candidate weights using random normal distribution
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        # Init output weights
        self.Wy = np.random.normal(size=(h, o))

        # Init biases as zeros
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step
        """
        # Concatenate hidden state and current input
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate (sigmoid)
        f_t = 1 / (1 + np.exp(-(np.dot(concat_input, self.Wf) + self.bf)))

        # Update/Input gate (sigmoid)
        u_t = 1 / (1 + np.exp(-(np.dot(concat_input, self.Wu) + self.bu)))

        # Candidate cell state (tanh)
        c_tilde = np.tanh(np.dot(concat_input, self.Wc) + self.bc)

        # Next cell state calculation
        c_next = f_t * c_prev + u_t * c_tilde

        # Output gate (sigmoid)
        o_t = 1 / (1 + np.exp(-(np.dot(concat_input, self.Wo) + self.bo)))

        # Next hidden state calculation
        h_next = o_t * np.tanh(c_next)

        # Output calculation (Softmax)
        y_linear = np.dot(h_next, self.Wy) + self.by
        exp_y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, c_next, y
