#!/usr/bin/env python3
"""
Normal distribution
"""


class Normal:
    """
    Normal distribution class
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialization function
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = sum(data) / len(data)
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        z-score function
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        x value function
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """
        pdf function
        """
        pi = 3.1415926536
        e = 2.7182818285

        a = (1 / (self.stddev * (2 * pi) ** 0.5))
        b = pow(e, (-(x - self.mean) ** 2) / (2 * (self.stddev ** 2)))
        return a * b

    def factorial(self, x):
        """
        factorial function
        """
        if x == 0:
            return 1

        fact = 1
        for i in range(1, x + 1):
            fact *= i
        return fact

    def erf(self, x):
        """
        erf function
        """
        pi = 3.1415926536
        summation = sum((pow(-1, n) * pow(x, (2 * n + 1)) /
                         (self.factorial(n) *
                          (2 * n + 1)) for n in range(5)))
        erf = (2 / (pi ** 0.5)) * summation
        return erf

    def cdf(self, x):
        """
        cdf function
        """
        erf = (x - self.mean) / (self.stddev * (2 ** 0.5))
        cdf = 0.5 * (1 + (self.erf(erf)))
        return cdf
