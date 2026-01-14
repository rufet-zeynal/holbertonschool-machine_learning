#!/usr/bin/env python3
"""
Math task
"""

def summation_i_squared(n):
    """
    Sum of square numbers
    """
    if type(n) is not int or n < 1:
        return None
    return n* (n + 1)* (2* n + 1)/6