#!/usr/bin/env python3
"""
Math task
"""


def poly_integral(poly, C=0):
    """
    Integrate a polynomial
    """
    if (type(poly) is not list or len(poly) == 0
            or not all(isinstance(i, (int, float)) for i in poly)
            or not isinstance(C, (int, float))):
        return None
    if len(poly) == 0:
        return None
    integral = [C] + [poly[i] / (i+1) for i in range(len(poly))]
    return integral
