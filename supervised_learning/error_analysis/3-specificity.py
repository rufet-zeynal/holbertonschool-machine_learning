#!/usr/bin/env python3
"""
Specificity calculation
"""
import numpy as np


def specificity(confusion):
    """
    Specificity of confusion matrix
    """
    # 1. The Diagonal: Correct "Yes" answers
    tp = np.diag(confusion)

    # 2. Total Actual: Sum of each row (True positives + False negatives)
    actual_pos = np.sum(confusion, axis=1)

    # 3. Total Predicted: Sum of each column (True positives + False positives)
    predicted_pos = np.sum(confusion, axis=0)

    # 4. Total Samples: Every single value in the matrix summed together
    total_samples = np.sum(confusion)

    # --- THE LOGIC ---

    # A. Actual Negatives: Everything in the data that is NOT the current class
    # (Total samples minus the row sum of the class)
    actual_neg = total_samples - actual_pos

    # B. False Positives: The "False Alarms"
    # (Everything we predicted as a class minus the ones we got right)
    fp = predicted_pos - tp

    # C. True Negatives: Actual Negatives minus the False Alarms
    tn = actual_neg - fp

    return tn / actual_neg
