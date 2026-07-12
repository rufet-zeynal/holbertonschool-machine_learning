#!/usr/bin/env python3
"""RNN Decoder module."""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNN Decoder class for machine translation."""

    def __init__(self, vocab, embedding, units, batch):
        """Initializes the decoder layers."""
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Decodes the encoded sequence to output words."""
        # Get the context vector from SelfAttention
        context, _ = self.attention(s_prev, hidden_states)

        # Get the embedding for the input target word x
        x_emb = self.embedding(x)

        # Expand context dimension from (batch, units) to (batch, 1, units)
        context_expanded = tf.expand_dims(context, 1)

        # Concatenate context vector and target word embedding in that order
        merged = tf.concat([context_expanded, x_emb], axis=-1)

        # Pass the merged representation into the GRU layer
        outputs, s = self.gru(merged, initial_state=s_prev)

        # Reshape or squeeze the GRU sequence output from (batch, 1, units)
        outputs = tf.squeeze(outputs, axis=1)

        # Get prediction logits for vocabulary distribution
        y = self.F(outputs)

        return y, s
