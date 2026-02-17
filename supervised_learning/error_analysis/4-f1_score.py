#!/usr/bin/env python3
"""
F1 score
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    F1 score
    """
    sensitivity = sensitivity(confusion)
    precision = precision(confusion)
    return 2 * (sensitivity * precision) / (sensitivity + precision)
