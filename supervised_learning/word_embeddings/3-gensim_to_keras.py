#!/usr/bin/env python3
"""Gensim to Keras"""
from tensorflow.keras.layers import Embedding


def gensim_to_keras(model):
    """Converts gensim word2vec to keras Embedding"""
    weights = model.wv.vectors
    return Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )
