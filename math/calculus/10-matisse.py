#!/usr/bin/env python3
"""
Math task
"""


def poly_derivative(poly):
    """
    Calculating the derivative of polynomial
    """
    if (type(poly) is not list or len(poly) == 0 or
            not all(isinstance(i, int) for i in poly)):
        return None
    if len(poly) == 1:
        return [0]
    return [i * poly[i] for i in range(1, len(poly))]