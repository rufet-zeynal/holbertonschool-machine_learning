#!/usr/bin/env python3
"""
Exponential distribution
"""


class Exponential:
    """
    class for exponential distribution
    """
    def __init__(self, data=None, lambtha=1.):
        """
        initialization
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must a positive value")
            self.lambtha = float(lambtha)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

        average = sum(data) / len(data)
        self.lambtha = 1 / average
