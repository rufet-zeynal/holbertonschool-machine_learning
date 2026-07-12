#!/usr/bin/env python3
"""RNN Decoder module for machine translation with attention"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decodes for machine translation, using a GRU and self attention"""

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor

        Args:
            vocab: integer, size of the output vocabulary
            embedding: integer, dimensionality of the embedding vector
            units: integer, number of hidden units in the RNN cell
            batch: integer, batch size
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Decodes for machine translation
        """
        context, _ = self.attention(s_prev, hidden_states)

        # embed the previous word -> shape (batch, 1, embedding)
        x = self.embedding(x)

        # give context a time dimension so it can concat with x
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)

        # run through the GRU, then drop the time dimension for the
        # dense layer
        outputs, s = self.gru(x)
        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))
        y = self.F(outputs)

        return y, s
