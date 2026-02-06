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

    def cdf(self, x):
        """
        cdf function
        """
        e = 2.7182818285
        x = float(x)
        value = (x - self.mean) / (self.stddev * (2 ** 0.5))

        t = 1 / (1 + 0.3275911 * abs(z))
        erf = 1 - (
                0.254829592 * t -
                0.284496736 * t ** 2 +
                1.421413741 * t ** 3 -
                1.453152027 * t ** 4 +
                1.061405429 * t ** 5
        ) * pow(e, -value ** 2)

        if z < 0:
            erf = -erf

        return 0.5 * (1 + erf)
