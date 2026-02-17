#!/usr/bin/env python3
"""
Precision calculation
"""
import numpy as np


def precision(confusion):
    """
    Precision of confusion matrix
    """
    true_positive_rate = np.diag(confusion)
    total_pred_pos = np.sum(confusion, axis=0)
    return true_positive_rate / total_pred_pos
