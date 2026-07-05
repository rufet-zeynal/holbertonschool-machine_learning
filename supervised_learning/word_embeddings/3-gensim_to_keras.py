#!/usr/bin/env python3
"""Gensim to Keras module"""
import tensorflow as tf


def gensim_to_keras(model):
    """Converts a gensim word2vec model to a keras Embedding layer"""
    # Extract the vocabulary size and embedding dimension
    vocab_size = model.wv.vectors.shape[0]
    vector_size = model.wv.vectors.shape[1]

    # Extract the actual weights from the model
    weights = [model.wv.vectors]

    # Create and return the trainable Keras Embedding layer
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=weights,
        trainable=True
    )

    return embedding_layer
