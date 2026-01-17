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
            or not isinstance(C, int)):
        return None

    integral = [C] + [poly[i] / (i+1) for i in range(len(poly))]
    # Convert whole floats to int
    integral = [int(x) if isinstance(x, float) and x.is_integer() else x
                for x in integral]

    # Remove trailing zeros
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
