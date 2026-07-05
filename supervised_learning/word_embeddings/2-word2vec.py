#!/usr/bin/env python3
"""creates and trains a gensim word2vec model"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                    negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """builds and trains a word2vec model"""
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window, min_count=min_count,
        workers=workers, seed=seed, sg=cbow,
        epochs=epochs, negative=negative)
    model.train(sentences=sentences, total_examples=model.corpus_count,
                epochs=model.epochs)
    return model
