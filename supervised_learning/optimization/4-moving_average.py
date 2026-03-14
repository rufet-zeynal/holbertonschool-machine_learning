#!/usr/bin/env python3
"""Moving Average"""


def moving_average(data, beta):
    """
    Calculateing weighted moving average of a data set
    data: list of data
    beta: weight used for the moving average
    """
    moving_average = []
    v = 0

    for t, x in enumerate(data, 1):
        v = beta * v + (1 - beta) * x
        v_corrected = v / (1 - beta ** t)
        moving_average.append(v_corrected)

    return moving_average
