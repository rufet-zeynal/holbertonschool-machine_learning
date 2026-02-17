#!/usr/bin/env python3
"""
Sensitivity calculation
"""


def sensitivity(confusion):
    """
    Sensitivity in a confusion matrix
    """
    true_positive = np.diag(confusion)

    actual_positive = np.sum(confusion, axis = 1)

    return true_positive / actual_positive
