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
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            average = sum(data) / len(data)
            self.lambtha = 1 / average

    def pdf(self, x):
        """
        pdf function
        """
        x = float(x)

        if x < 0:
            return 0

        e = 2.7182818285
        return (self.lambtha * pow(e, -self.lambtha * x))
