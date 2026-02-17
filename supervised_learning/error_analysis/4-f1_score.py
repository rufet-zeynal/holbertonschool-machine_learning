#!/usr/bin/env python3
"""
F1 score
"""
import numpy as np


def sensitivity(confusion):
    """
    Sensitivity in a confusion matrix
    """
    true_positive = np.diag(confusion)
    actual_positive = np.sum(confusion, axis=1)
    return (true_positive / actual_positive)

def precision(confusion):
    """
    Precision of confusion matrix
    """
    true_positive_rate = np.diag(confusion)
    total_pred_pos = np.sum(confusion, axis=0)
    return true_positive_rate / total_pred_pos

def f1_score(confusion):
    """
    F1 score
    """
    sensitivity = sensitivity(confusion)
    precision = precision(confusion)
    return 2 * (sensitivity * precision) / (sensitivity + precision)
