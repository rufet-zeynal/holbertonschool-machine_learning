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
    sens = sensitivity(confusion)
    prec = precision(confusion)
    f1 = 2 * (sens * prec) / (sens + prec)
    return f1
