#!/usr/bin/env python3
"""RNN Encoder module."""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNN Encoder class."""

    def __init__(self, vocab, embedding, units, batch):
        """Initializes the encoder."""
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """Initializes hidden states to zeros."""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Forward pass of the encoder."""
        embeddings = self.embedding(x)
        outputs, hidden = self.gru(embeddings, initial_state=initial)
        return outputs, hidden
