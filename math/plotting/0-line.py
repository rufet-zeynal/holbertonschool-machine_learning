#!/usr/bin/env python3
"""
Python code
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Line graph plotting
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8)
    plt.plot(y, 'r-')
    plt.xlim(0, 10)
    plt.show()
