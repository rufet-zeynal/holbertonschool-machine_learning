#!/usr/bin/env python3
"""
Trains a model with early stopping
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Training a model with using optional early stopping
    network         - the Keras model to train
    data            - input data, shape (m, nx)
    labels          - one-hot labels, shape (m, classes)
    batch_size      - size of each mini-batch
    epochs          - number of passes through data
    validation_data - tuple (X_val, Y_val) or None
    early_stopping  - whether to use early stopping
    patience        - epochs to wait before stopping
    verbose         - print output during training
    shuffle         - shuffle batches every epoch
    """
    callbacks = []
    if early_stopping and validation_data is not None:
        callbacks.append(
            K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            )
        )

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle
    )
    return history
