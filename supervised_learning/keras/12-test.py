#!/usr/bin/env python3
"""
Testing a neural network
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Test a neural network
    """
    return network.evaluate(
        data,
        labels,
        verbose=verbose
    )
