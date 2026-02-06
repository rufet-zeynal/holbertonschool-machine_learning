#!/usr/bin/env python3
"""
Binomial distribution
"""


class Binomial:
    """
    Binomial distribution class
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialization function
        """
        if data is None:
            if n < 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = n
            self.p = p

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)

        p = 1 - variance / mean
        n = round(mean / p)
        self.n = n
        self.p = p
