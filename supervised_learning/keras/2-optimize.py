#!/usr/bin/env python3
"""
Setting up Adam optimization for a Keras model
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Setting up ADAM optimization with categorical crossentropy
    for a Keras model
    network  - the Keras model to optimize
    alpha    - learning rate
    beta1    - first Adam parameter
    beta2    - second Adam parameter
    """
    network.compile(
        optimizer=K.optimizers.Adam(
            learning_rate=alpha,
            beta_1=beta1,
            beta_2=beta2
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
