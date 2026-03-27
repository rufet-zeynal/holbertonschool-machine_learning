#!/usr/bin/env python3
"""Learning rate decay using TensorFlow"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates inverse time decay learning rate schedule

    alpha      - original learning rate
    decay_rate - how fast alpha decays
    decay_step - steps before each decay

    Returns: learning rate schedule object
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True    # staircase=True means stepwise (not smooth)
    )
