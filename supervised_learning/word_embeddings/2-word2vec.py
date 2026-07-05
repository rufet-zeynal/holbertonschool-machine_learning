#!/usr/bin/env python3
"""Word2Vec module"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Creates, builds and trains a gensim word2vec model"""
    sg = 0 if cbow else 1

    # 1. Create the model
    model = gensim.models.Word2Vec(
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )

    # 2. Build the vocabulary
    model.build_vocab(sentences)

    # 3. Train the model
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
