#!/usr/bin/env python3
"""
Training a model with learning rate decay
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True,
                shuffle=False):
    """
    Training a model with learning rate decay
    network              - the Keras model to train
    data                 - input data, shape (m, nx)
    labels               - one-hot labels, shape (m, classes)
    batch_size           - size of each mini-batch
    epochs               - number of passes through data
    validation_data      - tuple (X_val, Y_val) or None
    early_stopping       - whether to use early stopping
    patience             - epochs to wait before stopping
    learning_rate_decay  - whether to use LR decay
    alpha                - initial learning rate
    decay_rate           - how fast the rate decays
    verbose              - print output during training
    shuffle              - shuffle batches every epoch
    """
    callbacks = []
    if early_stopping and validation_data is not None:
        callbacks.append(
            K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            )
        )

    if learning_rate_decay and validation_data is not None:
        def schedule(epoch):
            return alpha / (1 + decay_rate * epoch)

        callbacks.append(
            K.callbacks.LearningRateScheduler(
                schedule,
                verbose=verbose
            )
        )

    history = network.fit(
            data,
            labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=shuffle
        )
    return history
