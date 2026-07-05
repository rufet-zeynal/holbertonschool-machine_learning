#!/usr/bin/env python3
"""Word2Vec model training"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                    negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Builds and trains a gensim word2vec model"""
    # sg=0 means CBOW, sg=1 means skip-gram
    sg = 0 if cbow else 1

    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )

    # train the model for the given number of epochs
    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=epochs)

    return model
