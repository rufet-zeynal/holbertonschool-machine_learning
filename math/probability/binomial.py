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
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            p = 1 - (variance / mean)
            n = round(mean / p)
            p = mean / n
            self.n = n
            self.p = p

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

    def pmf(self, k):
        """
        Probability mass function
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        n_fact = self.factorial(self.n)
        k_fact = self.factorial(k)
        n_k_fact = self.factorial(self.n - k)
        a = n_fact / (k_fact * n_k_fact)

        b = pow(self.p, k) * pow(1 - self.p, self.n - k)
        return a * b

    def cdf(self, k):
        """
        cdf function
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        c_d_f = 0
        for i in range(1, k + 1):
            c_d_f += self.pmf(i)
        return c_d_f
