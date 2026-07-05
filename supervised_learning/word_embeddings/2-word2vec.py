#!/usr/bin/env python3
"""Word2Vec module"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Creates and trains a Word2Vec model"""
    return gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=0 if cbow else 1,
        epochs=epochs,
        seed=seed,
        workers=workers
    )
