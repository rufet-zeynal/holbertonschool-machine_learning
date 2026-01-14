#!/usr/bin/env python3


def summation_i_squared(n):
    if type(n) == int or n < 1:
        return None
    return n*(n+1)*(2*n+1)/6

n = int(input("n = "))
print(summation_i_squared(n))