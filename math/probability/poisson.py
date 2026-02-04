#!/usr/bin/env python3
"""
Poisson distribution
"""
import numpy as np


class Poisson:
    """
    Representing a Poisson distribution
    """
    def __init__(self, data=None, lambtha=1.):
        """
        Initializing a Poisson distribution
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

        else:
            if not isinstance(data, list):
                raise ValueError("data must be a list")

            if len(data) != 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = float(sum(data) / len(data))
