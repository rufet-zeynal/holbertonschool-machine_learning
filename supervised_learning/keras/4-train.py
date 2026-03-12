#!/usr/bin/env python3
"""Training a model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    Training a model using mini-batch gradient descent
    network    - the Keras model to train
    data       - input data, shape (m, nx)
    labels     - one-hot labels, shape (m, classes)
    batch_size - size of each mini-batch
    epochs     - number of passes through data
    verbose    - print output during training
    shuffle    - shuffle batches every epoch
    """
    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )
    return history
