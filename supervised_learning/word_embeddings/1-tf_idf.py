#!/usr/bin/env python3
"""TF-IDF module"""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def tf_idf(sentences, vocab=None):
    """Creates TF-IDF embedding"""
    vec = TfidfVectorizer(vocabulary=vocab)
    X = vec.fit_transform(sentences)

    # Handle different sklearn versions
    try:
        feat = vec.get_feature_names_out()
    except AttributeError:
        feat = vec.get_feature_names()

    return X.toarray(), np.array(feat)
