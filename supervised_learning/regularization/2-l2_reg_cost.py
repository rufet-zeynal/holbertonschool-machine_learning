#!/usr/bin/env python3
"""L2 Regularization Cost in Keras"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculating cost of neural network with L2 regularization
    cost:  tensor containing cost without L2 regularization
    model: Keras model with L2 regularization layers
    """
    return cost + model.losses
