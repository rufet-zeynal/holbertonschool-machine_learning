#!/usr/bin/env python3
"""BoW module"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """Creates BoW matrix"""
    if vocab is None:
        # Get unique words >= 2 chars
        vocab = sorted(set(w for s in sentences
                           for w in re.findall(r'\b\w\w+\b', s.lower())))

    # Init matrix
    E = np.zeros((len(sentences), len(vocab)), dtype=int)

    # Count words
    for i, s in enumerate(sentences):
        for w in re.findall(r'\b\w+\b', s.lower()):
            if w in vocab:
                E[i, vocab.index(w)] += 1

    return E, np.array(vocab)
