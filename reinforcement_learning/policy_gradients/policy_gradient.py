#!/usr/bin/env python3
"""Policy function for policy gradients"""
import numpy as np


def policy(matrix, weight):
    """Compute softmax policy from state matrix and weight matrix"""
    z = matrix.dot(weight)
    exp = np.exp(z - np.max(z))
    return exp / exp.sum(axis=1, keepdims=True)


def policy_gradient(state, weight):
    """Compute an action and its Monte-Carlo policy gradient"""
    state = state.reshape(1, -1)
    probs = policy(state, weight)
    action = np.random.choice(probs.shape[1], p=probs[0])

    s = probs.reshape(-1, 1)
    softmax_grad = np.diagflat(s) - s.dot(s.T)
    dlog = softmax_grad[action] / probs[0, action]
    gradient = state.T.dot(dlog.reshape(1, -1))
    return action, gradient
