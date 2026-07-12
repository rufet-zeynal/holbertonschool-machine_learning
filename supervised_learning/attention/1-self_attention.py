#!/usr/bin/env python3
"""Self Attention module."""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Self Attention class for machine translation."""

    def __init__(self, units):
        """Initializes the attention layers."""
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Calculates attention context vector and weights."""
        # Expand s_prev dimension to shape (batch, 1, units) for broadcasting
        w_s_prev = tf.expand_dims(self.W(s_prev), 1)

        # Apply the alignment model equation: tanh(W*s_prev + U*h)
        score = self.V(tf.nn.tanh(w_s_prev + self.U(hidden_states)))

        # Normalize score along alignment sequence length axis to get weights
        weights = tf.nn.softmax(score, axis=1)

        # Multiply weights by encoder states and sum to get context vector
        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)

        return context, weights
