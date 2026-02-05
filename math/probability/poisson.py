#!/usr/bin/env python3
"""
Poisson distribution
"""


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
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Probability mass function
        """
        try:
            k = int(k)
        except Exception:
            return 0

        if k < 0:
            return 0

        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        e = 2.7182818285
        return (pow(e, -self.lambtha) * pow(self.lambtha, k) / factorial)

    def cdf(self, k):
        """
        Cumulative distribution function
        """
        try:
            k = int(k)
        except Exception:
            return 0

        if k < 0:
            return 0

        cdf = 0
        for i in range(0, k + 1):
            cdf += self.pmf(i)

        return cdf
