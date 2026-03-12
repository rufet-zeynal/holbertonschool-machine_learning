#!/usr/bin/env python3
"""Converting a label vector into a one-hot matrix"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converting a label vector into a one-hot matrix
    labels - numpy array of labels
    classes - numpy array of classes
    """
    return K.utils.to_categorical(labels, classes)
